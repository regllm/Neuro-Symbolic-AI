use crate::poker::{card, math};

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use std::collections::HashSet;


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
    deck.iter().combinations(count)
        .map(|c| c.into_iter().cloned().collect())
        .collect()
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

    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(max_count as u64);
    progress.set_style(style.clone());

    for start_combo in start_combos {
        let start_combo_set: HashSet<i32> = start_combo
            .clone()
            .into_iter()
            .collect();
        let available_cards: Vec<&i32> = deck
            .into_iter()
            .filter(|x| !start_combo_set.contains(x))
            .collect();
        let publics = available_cards
            .into_iter()
            .cloned()
            .combinations(public_count);

        for public_combo in publics {
            let mut combo: Vec<i32> = Vec::with_capacity(hand_size);
            for &x in start_combo.iter().rev() {
                combo.push(x);
            }
            for &x in public_combo.iter().rev() {
                combo.push(x);
            }
            result_combos.push(combo);
            progress.inc(1);
        }
    }
    progress.finish();

    result_combos
}
