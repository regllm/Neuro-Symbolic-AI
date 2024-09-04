mod combo;
mod card;
mod math;

use std::time::{Instant, Duration};


fn main() {
    // let result = card::new(14, 's');
    // println!("Card representation for Spade A: {}", result);

    let deck = combo::create_deck(2, 4);
    
    // let deck = combo::create_deck(2, 14);
    let start_combos = combo::create_card_combos(&deck, 2);

    // let flop_start_time = Instant::now();
    // let flop_combos = combo::create_info_combos(&deck, &start_combos, 3);
    // let flop_end_time = Instant::now();
    // let flop_elapsed_time = flop_end_time - flop_start_time;
    // println!("Created Flop combos in {:?} seconds.", flop_elapsed_time);

    // let turn_start_time = Instant::now();
    // let turn_combos = combo::create_info_combos(&deck, &start_combos, 4);
    // let turn_end_time = Instant::now();
    // let turn_elapsed_time = turn_end_time - turn_start_time;
    // println!("Created Turn combos in {:?} seconds.", turn_elapsed_time);

    let river_start_time = Instant::now();
    let river_combos = combo::create_info_combos(&deck, &start_combos, 5);
    let river_end_time = Instant::now();
    let river_elapsed_time = river_end_time - river_start_time;
    println!("Created River combos in {:?} seconds.", river_elapsed_time);

    let lookup = card::load_lookup("./assets/lookup.json");

    // let combo: Vec<i32> = vec![
    //     card::new(14, 's'),
    //     card::new(13, 's'),
    //     card::new(12, 's'),
    //     card::new(11, 's'),
    //     card::new(10, 's'),
    // ];

    // let combo: Vec<i32> = vec![
    //     card::new(14, 's'),
    //     card::new(11, 's'),
    //     card::new(11, 'h'),
    //     card::new(10, 's'),
    //     card::new(8, 's'),
    // ];

    // let score = card::evaluate(&combo, &lookup);
    // println!("{:?}", score);

    let result = combo::simulate_river_hand_strengths(&deck, &river_combos, &lookup, 3);
    println!("{:?}", result);
}
