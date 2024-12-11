mod cluster;
mod eval;
mod math;

use std::time::{Instant, Duration};


fn main() {
    // let result = eval::new(14, 's');
    // println!("Card representation for Spade A: {}", result);

    let deck = cluster::create_deck(2, 5);
    let start_combos = cluster::create_card_combos(&deck, 2);

    let flop_start_time = Instant::now();
    let flop_combos = cluster::create_info_combos(&deck, &start_combos, 3);
    let flop_end_time = Instant::now();
    let flop_elapsed_time = flop_end_time - flop_start_time;
    println!("Created Flop combos in {:?} seconds.", flop_elapsed_time);

    let turn_start_time = Instant::now();
    let turn_combos = cluster::create_info_combos(&deck, &start_combos, 4);
    let turn_end_time = Instant::now();
    let turn_elapsed_time = turn_end_time - turn_start_time;
    println!("Created Turn combos in {:?} seconds.", turn_elapsed_time);

    let river_start_time = Instant::now();
    let river_combos = cluster::create_info_combos(&deck, &start_combos, 5);
    let river_end_time = Instant::now();
    let river_elapsed_time = river_end_time - river_start_time;
    println!("Created River combos in {:?} seconds.", river_elapsed_time);
}