use crate::card;
use crate::distance;
use crate::shuffle;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;


fn simulate_river_games(
    deck: &Vec<i32>,
    river_combo: &Vec<i32>,
    lookup: &card::LookupTable,
    river_simulation_count: u32,
) -> Vec<f64> {
    let base_rank = card::evaluate(river_combo, lookup);
    let available_cards: Vec<&i32> = deck.iter()
        .filter(|&x| !river_combo.contains(x))
        .collect();
    let available_cards_count = available_cards.len();
    
    let prob_unit = 1.0 / f64::from(river_simulation_count);
    let mut result: Vec<f64> = vec![0.0f64, 0.0f64, 0.0f64];

    let mut another_combo: Vec<i32> = river_combo.to_vec();
    let mut rng = rand::thread_rng();
    for _i in 0..river_simulation_count {
        let (r1, r2) = shuffle::get_random_index_pair(&mut rng, available_cards_count);
        another_combo[0] = *available_cards[r1];
        another_combo[1] = *available_cards[r2];

        let another_rank = card::evaluate(&another_combo, lookup);
        if base_rank > another_rank {
            result[0] += prob_unit;
        } else if base_rank < another_rank {
            result[1] += prob_unit;
        } else {
            result[2] += prob_unit;
        }
    }

    result
}

pub fn simulate_river_hand_strengths(
    deck: &Vec<i32>,
    river_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_simulation_count: u32,
) -> Vec<Vec<f64>> {
    let river_combos_size = river_combos.len();
    let result_width = 3;

    let mut result: Vec<Vec<f64>> =
        Vec::with_capacity(river_combos_size);

    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(river_combos_size as u64);
    progress.set_style(style.clone());

    for river_combo in river_combos {
        result.push(
            simulate_river_games(
                deck,
                &river_combo,
                lookup,
                river_simulation_count,
            )
        );
        progress.inc(1);
    }
    progress.finish();

    result
}

fn simulate_turn_ehs_distributions(
    deck: &Vec<i32>,
    turn_combo: &Vec<i32>,
    river_centroids: &Vec<Vec<f64>>,
    lookup: &card::LookupTable,
    river_simulation_count: u32,
    turn_simulation_count: u32,
    river_cluster_count: u32,
) -> Vec<f64> {
    let mut available_cards: Vec<&i32> = deck.iter()
        .filter(|&x| !turn_combo.contains(x))
        .collect();
    let available_cards_count = available_cards.len();
    
    let prob_unit = 1.0 / f64::from(river_simulation_count);
    let sub_prob_unit = 1.0 / f64::from(turn_simulation_count);
    let mut ehs: Vec<f64> = vec![0.0, 0.0, 0.0];

    let mut base_combo: Vec<i32> = turn_combo.to_vec();
    base_combo.push(0);
    let mut another_combo: Vec<i32> = base_combo.to_vec();

    let mut result: Vec<f64> = Vec::with_capacity(river_cluster_count as usize);
    for _i in 0..river_cluster_count {
        result.push(0.0);
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
                ehs[0] += prob_unit;
            } else if base_rank < another_rank {
                ehs[1] += prob_unit;
            } else {
                ehs[2] += prob_unit;
            }
        }

        // Get EMD for expected hand strength against each river centroid
        // to which does it belong?
        let mut min_centroid_index: usize = 0;
        let mut min_dist: f64 = -1.0;
        for (i, river_centroid) in river_centroids.iter().enumerate() {
            let dist = distance::wasserstein(&ehs, &river_centroid);
            if min_dist < 0.0 {
                min_dist = dist;
            } else if dist < min_dist {
                min_centroid_index = i;
                min_dist = dist;
            }
        }
        result[min_centroid_index] += prob_unit;
    }

    result
}

pub fn simulate_turn_hand_strengths(
    deck: &Vec<i32>,
    turn_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_simulation_count: u32,
    turn_simulation_count: u32,
    river_cluster_count: u32,
) -> Vec<Vec<f64>> {
    let turn_combos_size = turn_combos.len();
    let result_width = river_cluster_count;

    let mut result: Vec<Vec<f64>> = Vec::with_capacity(turn_combos_size);

    for turn_combo in turn_combos {
        result.push(
            simulate_river_games(
                deck,
                &turn_combo,
                lookup,
                river_simulation_count,
            )
        )
    }

    result
}
