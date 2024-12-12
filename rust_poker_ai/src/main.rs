mod card;
mod cluster;
mod combo;
mod distance;
mod math;
mod shuffle;
mod strength;

use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Instant, Duration};
use std::fs::File;
use std::io::{self, Write};
use rand::seq::SliceRandom;
use ndarray::array;


const TURN_SIMULATION_COUNT: u32 = 6;
const RIVER_SIMULATION_COUNT: u32 = 6;
const RIVER_CLUSTER_COUNT: u32 = 50;


fn test() {
    // let target: Vec<Vec<f64>> = vec![
    //     vec![0.0f64, 0.1f64, 0.0f64, 0.0f64],
    //     vec![0.1f64, 0.1f64, 0.0f64, 0.0f64],
    //     vec![0.0f64, 0.1f64, 0.1f64, 0.0f64],
    //     vec![2.0f64, 2.1f64, 2.0f64, 0.0f64],
    //     vec![2.1f64, 2.1f64, 2.0f64, 0.0f64],
    //     vec![2.0f64, 2.1f64, 2.1f64, 0.0f64],
    // ];

    // let encoded: Vec<u8> = bincode::serialize(&target).unwrap();

    // let mut file = File::create(filename);
    // file.write_all(&encoded);

    // let decoded: Vec<Vec<f64>> = bincode::deserialize(&encoded[..]).unwrap();
    // assert_eq!(target, decoded);
    
    
    // let result = card::new(14, 's');
    // println!("Card representation for Spade A: {}", result);

    let p = vec![0.1, 0.2, 0.3, 0.4];
    let q = vec![0.2, 0.3, 0.2, 0.3];

    // Calculate Wasserstein distance
    let distance = distance::wasserstein(&p, &q);
    println!("Wasserstein distance: {}", distance);

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

    let (centers, clusters) = cluster::kmeans(
        &result,
        RIVER_CLUSTER_COUNT,
    );

    println!("Kmeans centers: {:?}", centers);
    println!("Kmeans clusters: {:?}", clusters);
}

fn build_river_lut(
    deck: &Vec<i32>,
    start_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
) {
    println!("Creating River combos.");
    let mut start_time = Instant::now();
    let river_combos = combo::create_info_combos(&deck, &start_combos, 5);
    let mut elapsed_time = Instant::now() - start_time;
    println!("Created River combos in {:?}.", elapsed_time);

    println!("Simulating River hand strengths.");
    start_time = Instant::now();
    let result = strength::simulate_river_hand_strengths(
        deck,
        &river_combos,
        lookup,
        RIVER_SIMULATION_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Simulated River hand strengths in {:?}.", elapsed_time);

    // println!("River hand strengths: {:?}.", result);

    for row in result.iter() {
        if row.len() != 3 {
            println!("There is a row with length {:?}.", row.len());
        }
    }

    println!("Clustering River combos.");
    start_time = Instant::now();
    let (centroids, clusters) = cluster::kmeans(
        &result,
        RIVER_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Clustered River combos in {:?}.", elapsed_time);

    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();

    println!("Writing River centroids.");
    let write_centroid_progress = ProgressBar::new(centroids.len() as u64);
    write_centroid_progress.set_style(style.clone());
    start_time = Instant::now();
    let mut centroids_file = File::create("./output/river_centroids.txt").unwrap();
    for row in centroids.iter() {
        let row_str = row.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(",");
        writeln!(centroids_file, "{}", row_str).unwrap();
        write_centroid_progress.inc(1);
    }
    write_centroid_progress.finish();
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River centroids in {:?}.", elapsed_time);
    
    println!("Writing River clusters.");
    let write_clusters_progress = ProgressBar::new(clusters.len() as u64);
    write_clusters_progress.set_style(style.clone());
    start_time = Instant::now();
    let mut clusters_file = File::create("./output/river_clusters.txt").unwrap();
    for row in clusters.iter() {
        writeln!(clusters_file, "{}", row.to_string()).unwrap();
        write_clusters_progress.inc(1);
    }
    write_clusters_progress.finish();
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River clusters in {:?}.", elapsed_time);
}

fn build_lut() {
    // let deck = combo::create_deck(2, 4);
    let deck = combo::create_deck(2, 14);
    let start_combos = combo::create_card_combos(&deck, 2);
    let lookup = card::load_lookup("./assets/lookup.json");

    build_river_lut(&deck, &start_combos, &lookup);
}


fn main() {
    // test();
    build_lut();
}
