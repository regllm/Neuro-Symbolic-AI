use crate::cluster::{distance, util, shuffle};
use crate::poker::card;
use crate::progress;

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rayon::prelude::*;
use std::cmp;
use std::mem::drop;
use std::thread::available_parallelism;


fn calc_chunk_size(total: usize) -> usize {
    let process_count = available_parallelism().unwrap().get();
    let chunk_size = cmp::min((total as f64 / process_count as f64).ceil() as usize, 10000 as usize);
    chunk_size
}

fn simulate_river_games(
    deck: &Vec<i32>,
    river_combo: &Vec<i32>,
    lookup: &card::LookupTable,
    river_simulation_count: u32,
) -> Vec<u8> {
    let base_rank = card::evaluate(river_combo, lookup);
    let available_cards: Vec<&i32> = deck.iter()
        .filter(|&x| !river_combo.contains(x))
        .collect();
    let available_cards_count = available_cards.len();
    
    // let prob_unit = 1.0 / f64::from(river_simulation_count);
    let mut result: Vec<u8> = vec![0, 0, 0];

    let mut another_combo: Vec<i32> = river_combo.to_vec();
    let mut rng = rand::thread_rng();
    for _i in 0..river_simulation_count {
        let (r1, r2) = shuffle::get_random_index_pair(&mut rng, available_cards_count);
        another_combo[0] = *available_cards[r1];
        another_combo[1] = *available_cards[r2];

        let another_rank = card::evaluate(&another_combo, lookup);
        if base_rank > another_rank {
            result[0] += 1;
        } else if base_rank < another_rank {
            result[1] += 1;
        } else {
            result[2] += 1;
        }
    }

    result
}

pub fn simulate_river_hand_strengths(
    deck: &Vec<i32>,
    river_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_simulation_count: u32,
) -> Vec<Vec<u8>> {
    let river_combos_size = river_combos.len();

    let mut result: Vec<Vec<u8>> =
        Vec::with_capacity(river_combos_size);

    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(river_combos_size as u64);
    progress.set_style(style);

    let chunk_size = calc_chunk_size(river_combos_size);
    for chunk in &river_combos.into_iter().chunks(chunk_size) {
        let chunk_clone: Vec<Vec<i32>> = chunk.cloned().collect();
        let chunk_result: Vec<Vec<u8>> = chunk_clone.par_iter()
            .map(|river_combo| {
                simulate_river_games(
                    deck,
                    &river_combo,
                    lookup,
                    river_simulation_count,
                )
            })
            .collect();
        let chunk_size = chunk_result.len() as u64;
        result.extend(chunk_result);
        progress.inc(chunk_size);
    }
    progress.finish();

    result
}

fn simulate_turn_ehs_distributions(
    deck: &Vec<i32>,
    turn_combo: &Vec<i32>,
    lookup: &card::LookupTable,
    emd_calc: &mut distance::EmdCalculator,
    river_centroids: &Vec<Vec<f64>>,
    river_simulation_count: u32,
    turn_simulation_count: u32,
    river_cluster_count: u32,
) -> Vec<u8> {
    let available_cards: Vec<&i32> = deck.iter()
        .filter(|&x| !turn_combo.contains(x))
        .collect();
    let available_cards_count = available_cards.len();
    
    let mut ehs: Vec<u8> = vec![0, 0, 0];

    let mut base_combo: Vec<i32> = turn_combo.to_vec();
    base_combo.push(0);
    let mut another_combo: Vec<i32> = base_combo.to_vec();

    let mut result: Vec<u8> = Vec::with_capacity(river_cluster_count as usize);
    for _i in 0..river_cluster_count {
        result.push(0);
    }

    // Sample river cards and run simulations.
    let mut rng = rand::thread_rng();
    for _i in 0..turn_simulation_count {
        let r = shuffle::get_random_index(&mut rng, available_cards_count);
        base_combo[6] = *available_cards[r];
        another_combo[6] = *available_cards[r];

        for _j in 0..river_simulation_count {
            let (r1, r2) = shuffle::get_random_index_pair_with_except(
                &mut rng,
                available_cards_count,
                r,
            );
            another_combo[0] = *available_cards[r1];
            another_combo[1] = *available_cards[r2];

            let base_rank = card::evaluate(&base_combo, lookup);
            let another_rank = card::evaluate(&another_combo, lookup);
            if base_rank > another_rank {
                ehs[0] += 1;
            } else if base_rank < another_rank {
                ehs[1] += 1;
            } else {
                ehs[2] += 1;
            }
        }

        // Get EMD for expected hand strength against each river centroid
        // to which does it belong?
        let mut min_centroid_index: usize = 0;
        let mut min_dist: f64 = -1.0;
        for (i, river_centroid) in river_centroids.iter().enumerate() {
            let ehs_f64: Vec<f64> = ehs.iter().map(|&x| x as f64).collect();
            let dist = emd_calc.calc(&ehs_f64, &river_centroid).expect("EMD calculation failed");
            // let dist = 0.5;
            if min_dist < 0.0 {
                min_dist = dist;
            } else if dist < min_dist {
                min_centroid_index = i;
                min_dist = dist;
            }
        }
        // now increment the cluster to which it belongs -
        result[min_centroid_index] += 1;
        // unsafe {
        //     let mut_result: *mut u8 = result as *mut u8;
        //     *mut_result.offset(min_centroid_index.try_into().unwrap()) += 1;
        // }
    }

    result
}

pub fn simulate_turn_hand_strengths(
    deck: &Vec<i32>,
    turn_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_centroids: &Vec<Vec<f64>>,
    river_simulation_count: u32,
    turn_simulation_count: u32,
    river_cluster_count: u32,
) -> Vec<Vec<u8>> {
    let turn_combos_size = turn_combos.len();

    // Initialize the result vector.
    let mut result: Vec<Vec<u8>> = Vec::with_capacity(turn_combos_size);
    // let mut shared_result: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::with_capacity(turn_combos_size)));
    // for i in 0..turn_combos_size {
    //     let mut row: Vec<u8> = Vec::with_capacity(river_cluster_count as usize);
    //     for _j in 0..river_cluster_count {
    //         row.push(0u8);
    //     }
    //     result.push(row);
    //     // shared_result.lock().unwrap().push(row);
    // }

    let progress = progress::new(turn_combos_size as u64);

    let chunk_size = calc_chunk_size(turn_combos_size);
    for chunk in &turn_combos.into_iter().chunks(chunk_size) {
        let chunk_clone: Vec<Vec<i32>> = chunk.cloned().collect();
        let curr_chunk_size = chunk_clone.len() as u64;

        let chunk_result: Vec<Vec<u8>> = chunk_clone.par_iter()
            .map_init(
                || distance::EmdCalculator::new(),
                |emd_calc, turn_combo| {
                    simulate_turn_ehs_distributions(
                        deck,
                        &turn_combo,
                        lookup,
                        emd_calc,
                        river_centroids,
                        river_simulation_count,
                        turn_simulation_count,
                        river_cluster_count,
                    )
                },
            )
            .collect();
        
        for combo in chunk_clone {
            drop(combo);
        }
        result.extend(chunk_result);
        
        progress.inc(curr_chunk_size);
    }
    progress.finish();

    // Clone result values row by row to get pure result vectors.
    // let raw_result = shared_result.lock().unwrap();
    // let result_vector: &Vec<Vec<u8>> = &*raw_result;

    // let mut result: Vec<Vec<u8>> = Vec::new();
    // for row in result_vector {
    //     result.push(row.clone());
    // }
    result
}

fn simulate_flop_potential_aware_distributions(
    deck: &Vec<i32>,
    flop_combo: &Vec<i32>,
    lookup: &card::LookupTable,
    emd_calc: &mut distance::EmdCalculator,
    river_centroids: &Vec<Vec<f64>>,
    turn_centroids: &Vec<Vec<f64>>,
    river_simulation_count: u32,
    turn_simulation_count: u32,
    flop_simulation_count: u32,
    river_cluster_count: u32,
    turn_cluster_count: u32,
) -> Vec<u8> {
    let available_cards: Vec<&i32> = deck.iter()
        .filter(|&x| !flop_combo.contains(x))
        .collect();
    let available_cards_count = available_cards.len();
    
    // let prob_unit = 1.0 / f64::from(flop_simulation_count);

    let mut augmented_turn_combo: Vec<i32> = flop_combo.to_vec();
    augmented_turn_combo.push(0);

    let mut result: Vec<u8> = Vec::with_capacity(turn_cluster_count as usize);
    for _i in 0..turn_cluster_count {
        result.push(0);
    }

    let mut rng = rand::thread_rng();
    for _i in 0..flop_simulation_count {
        // Randomly generating a Turn card.
        let r = shuffle::get_random_index(&mut rng, available_cards_count);
        augmented_turn_combo[5] = *available_cards[r];

        // let mut turn_ehs_distribution: Vec<u8> = Vec::with_capacity(river_cluster_count as usize);
        // for _i in 0..river_cluster_count {
        //     turn_ehs_distribution.push(0);
        // }
        let turn_ehs_distribution = simulate_turn_ehs_distributions(
            deck,
            &augmented_turn_combo,
            lookup,
            emd_calc,
            river_centroids,
            river_simulation_count,
            turn_simulation_count,
            river_cluster_count,
        );

        let mut min_centroid_index: usize = 0;
        let mut min_dist: f64 = -1.0;
        for (j, turn_centroid) in turn_centroids.iter().enumerate() {
            let ted_f64: Vec<f64> = turn_ehs_distribution.iter().map(|&x| x as f64).collect();
            let dist = emd_calc.calc(&ted_f64, &turn_centroid).expect("EMD calculation failed");
            if min_dist < 0.0 {
                min_dist = dist;
            } else if dist < min_dist {
                min_centroid_index = j;
                min_dist = dist;
            }
        }
        result[min_centroid_index] += 1;
    }

    result
}

pub fn simulate_flop_hand_strengths(
    deck: &Vec<i32>,
    flop_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_centroids: &Vec<Vec<f64>>,
    turn_centroids: &Vec<Vec<f64>>,
    river_simulation_count: u32,
    turn_simulation_count: u32,
    flop_simulation_count: u32,
    river_cluster_count: u32,
    turn_cluster_count: u32,
) -> Vec<Vec<u8>> {
    let flop_combos_size = flop_combos.len();
    // let result_width = river_cluster_count;

    let mut result: Vec<Vec<u8>> = Vec::with_capacity(flop_combos_size);

    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(flop_combos_size as u64);
    progress.set_style(style);

    let chunk_size = calc_chunk_size(flop_combos_size);
    for chunk in &flop_combos.into_iter().chunks(chunk_size) {
        let chunk_clone: Vec<Vec<i32>> = chunk.cloned().collect();

        let chunk_result: Vec<Vec<u8>> = chunk_clone.par_iter()
            .map_init(
                || distance::EmdCalculator::new(),
                |emd_calc, flop_combo| {
                    simulate_flop_potential_aware_distributions(
                        deck,
                        &flop_combo,
                        lookup,
                        emd_calc,
                        river_centroids,
                        turn_centroids,
                        river_simulation_count,
                        turn_simulation_count,
                        flop_simulation_count,
                        river_cluster_count,
                        turn_cluster_count,
                    )
                },
            )
            .collect();
        let chunk_size = chunk_result.len() as u64;
        result.extend(chunk_result);
        progress.inc(chunk_size);
    }
    progress.finish();

    result
}
