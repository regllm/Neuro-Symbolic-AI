use crate::card;
use crate::math;
use std::collections::HashSet;
use itertools::Itertools;
use rand::seq::SliceRandom;


pub fn create_deck(low_rank: u8, high_rank: u8) -> Vec<i32> {
    let base_suits = ['s', 'h', 'd', 'c'];

    let rank_count = (high_rank - low_rank + 1) as usize;
    let mut deck: Vec<i32> = Vec::with_capacity(rank_count * 4);

    for &suit in &base_suits {
        for rank in low_rank..=high_rank {
            deck.push(card::new(rank, suit));
        }
    }

    deck.sort();

    deck
}

pub fn create_card_combos(deck: &Vec<i32>, count: usize) -> Vec<Vec<i32>> {
    deck.iter().combinations(count).map(|c| c.into_iter().cloned().collect()).collect()
}

pub fn create_info_combos(
    deck: &Vec<i32>,
    start_combos: &Vec<Vec<i32>>,
    public_count: usize,
) -> Vec<Vec<i32>> {
    let max_count = start_combos.len()
        * math::count_combinations(
            deck.len() - start_combos[0].len(),
            public_count,
        );
    let hand_size = start_combos[0].len() + public_count;
    let mut result_combos: Vec<Vec<i32>> = Vec::with_capacity(max_count);

    for start_combo in start_combos {
        let start_combo_set: HashSet<i32> = start_combo.clone().into_iter().collect();
        let available_cards: Vec<&i32> = deck.into_iter().filter(|x| !start_combo_set.contains(x)).collect();
        let publics = available_cards.into_iter().cloned().combinations(public_count);

        for public_combo in publics {
            let mut combo: Vec<i32> = Vec::with_capacity(hand_size);
            for &x in start_combo.iter().rev() {
                combo.push(x);
            }
            for &x in public_combo.iter().rev() {
                combo.push(x);
            }
            result_combos.push(combo);
        }
    }

    result_combos
}

fn simulate_river_games(
    deck: &Vec<i32>,
    river_combo: &Vec<i32>,
    lookup: &card::LookupTable,
    simulation_count: u16,
) -> Vec<f32> {
    let base_rank = card::evaluate(river_combo, lookup);
    let mut available_cards: Vec<&i32> = deck.iter()
        .filter(|&x| !river_combo.contains(x))
        .collect();
    
    let prob_unit = 1.0 / f32::from(simulation_count);
    let mut result: Vec<f32> = vec![0.0, 0.0, 0.0];

    let mut another_combo: Vec<i32> = river_combo.to_vec();
    for _i in 0..simulation_count {
        let mut rng = rand::thread_rng();
        available_cards.shuffle(&mut rng);
        another_combo[0] = *available_cards[0];
        another_combo[1] = *available_cards[1];

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
    simulation_count: u16,
) -> Vec<Vec<f32>> {
    let river_combos_size = river_combos.len();
    let result_width = 3;

    let mut result: Vec<Vec<f32>> = Vec::with_capacity(river_combos_size);

    for river_combo in river_combos {
        result.push(simulate_river_games(deck, &river_combo, lookup, simulation_count))
    }

    result
}
