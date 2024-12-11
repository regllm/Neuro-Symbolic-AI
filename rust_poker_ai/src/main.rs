mod combo;
mod card;
mod distance;
mod math;
mod shuffle;
mod strength;

use std::time::{Instant, Duration};
use rand::seq::SliceRandom;
use ndarray::array;


const TURN_SIMULATION_COUNT: u16 = 6;
const RIVER_SIMULATION_COUNT: u16 = 6;


fn main() {
    // let result = card::new(14, 's');
    // println!("Card representation for Spade A: {}", result);

    // let p = vec![0.1, 0.2, 0.3, 0.4];
    // let q = vec![0.2, 0.3, 0.2, 0.3];

    // // Calculate Wasserstein distance
    // let distance = distance::wasserstein(&p, &q);

    // println!("Wasserstein distance: {}", distance);

    let deck = combo::create_deck(2, 4);
    
    // let deck = combo::create_deck(2, 14);
    let start_combos = combo::create_card_combos(&deck, 2);

    // let flop_start_time = Instant::now();
    // let flop_combos = combo::create_info_combos(&deck, &start_combos, 3);
    // let flop_end_time = Instant::now();
    // let flop_elapsed_time = flop_end_time - flop_start_time;
    // println!("Created Flop combos in {:?} seconds.", flop_elapsed_time);

    let turn_start_time = Instant::now();
    let turn_combos = combo::create_info_combos(&deck, &start_combos, 4);
    let turn_end_time = Instant::now();
    let turn_elapsed_time = turn_end_time - turn_start_time;
    println!("Created Turn combos in {:?}.", turn_elapsed_time);

    let river_start_time = Instant::now();
    let river_combos = combo::create_info_combos(&deck, &start_combos, 5);
    let river_end_time = Instant::now();
    let river_elapsed_time = river_end_time - river_start_time;
    println!("Created River combos in {:?}.", river_elapsed_time);

    let lookup = card::load_lookup("./assets/lookup.json");

    let river_simulate_start_time = Instant::now();
    let result = strength::simulate_river_hand_strengths(
        &deck,
        &river_combos,
        &lookup,
        RIVER_SIMULATION_COUNT,
    );
    let river_simulate_end_time = Instant::now();
    let river_simulate_elapsed_time = 
        river_simulate_end_time - river_simulate_start_time;
    println!("Simulated River hand strengths in {:?}.", river_simulate_elapsed_time);
}
