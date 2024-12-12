mod card;
mod cluster;
mod combo;
mod distance;
mod math;
mod shuffle;
mod strength;
// mod test;

use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Instant, Duration};
use std::fs::File;
use std::io::{self, Write};


const FLOP_SIMULATION_COUNT: u32 = 6;
const TURN_SIMULATION_COUNT: u32 = 6;
const RIVER_SIMULATION_COUNT: u32 = 6;
const FLOP_CLUSTER_COUNT: u32 = 50;
const TURN_CLUSTER_COUNT: u32 = 50;
const RIVER_CLUSTER_COUNT: u32 = 50;


fn save_combos(combos: &Vec<Vec<i32>>, file_path: &str) {
    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(combos.len() as u64);
    progress.set_style(style);

    let mut file = File::create(file_path).unwrap();
    for row in combos.iter() {
        let row_str = row.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(",");
        writeln!(file, "{}", row_str).unwrap();
        progress.inc(1);
    }

    progress.finish();
}


fn save_centroids(centroids: &Vec<Vec<f64>>, file_path: &str) {
    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(centroids.len() as u64);
    progress.set_style(style);

    let mut file = File::create(file_path).unwrap();
    for row in centroids.iter() {
        let row_str = row.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(",");
        writeln!(file, "{}", row_str).unwrap();
        progress.inc(1);
    }

    progress.finish();
}


fn save_clusters(clusters: &Vec<u32>, file_path: &str) {
    let style = ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap();
    let progress = ProgressBar::new(clusters.len() as u64);
    progress.set_style(style);

    let mut file = File::create(file_path).unwrap();
    for row in clusters.iter() {
        writeln!(file, "{}", row.to_string()).unwrap();
        progress.inc(1);
    }

    progress.finish();
}


fn build_river_lut(
    deck: &Vec<i32>,
    start_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
) -> (Vec<Vec<f64>>, Vec<u32>) {
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

    println!("Writing River combos.");
    start_time = Instant::now();
    save_combos(&river_combos, "./output/river_combos.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River combos in {:?}.", elapsed_time);
    
    println!("Writing River centroids.");
    start_time = Instant::now();
    save_centroids(&centroids, "./output/river_centroids.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River centroids in {:?}.", elapsed_time);
    
    println!("Writing River clusters.");
    start_time = Instant::now();
    save_clusters(&clusters, "./output/river_clusters.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River clusters in {:?}.", elapsed_time);

    (centroids, clusters)
}


fn build_turn_lut(
    deck: &Vec<i32>,
    start_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_centroids: &Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    println!("Creating Turn combos.");
    let mut start_time = Instant::now();
    let turn_combos = combo::create_info_combos(&deck, &start_combos, 4);
    let mut elapsed_time = Instant::now() - start_time;
    println!("Created Turn combos in {:?}.", elapsed_time);

    println!("Simulating Turn hand strengths.");
    start_time = Instant::now();
    let result = strength::simulate_turn_hand_strengths(
        deck,
        &turn_combos,
        lookup,
        river_centroids,
        RIVER_SIMULATION_COUNT,
        TURN_SIMULATION_COUNT,
        RIVER_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Simulated Turn hand strengths in {:?}.", elapsed_time);

    println!("Clustering Turn combos.");
    start_time = Instant::now();
    let (centroids, clusters) = cluster::kmeans(
        &result,
        TURN_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Clustered Turn combos in {:?}.", elapsed_time);

    println!("Writing Turn combos.");
    start_time = Instant::now();
    save_combos(&turn_combos, "./output/turn_combos.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Turn combos in {:?}.", elapsed_time);

    println!("Writing Turn centroids.");
    start_time = Instant::now();
    save_centroids(&centroids, "./output/turn_centroids.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Turn centroids in {:?}.", elapsed_time);
    
    println!("Writing Turn clusters.");
    start_time = Instant::now();
    save_clusters(&clusters, "./output/turn_clusters.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Turn clusters in {:?}.", elapsed_time);

    (centroids, clusters)
}


fn build_flop_lut(
    deck: &Vec<i32>,
    start_combos: &Vec<Vec<i32>>,
    lookup: &card::LookupTable,
    river_centroids: &Vec<Vec<f64>>,
    turn_centroids: &Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<u32>) {
    println!("Creating Flop combos.");
    let mut start_time = Instant::now();
    let flop_combos = combo::create_info_combos(&deck, &start_combos, 3);
    let mut elapsed_time = Instant::now() - start_time;
    println!("Created Flop combos in {:?}.", elapsed_time);

    println!("Simulating Turn hand strengths.");
    start_time = Instant::now();
    let result = strength::simulate_flop_hand_strengths(
        deck,
        &flop_combos,
        lookup,
        river_centroids,
        turn_centroids,
        RIVER_SIMULATION_COUNT,
        TURN_SIMULATION_COUNT,
        FLOP_SIMULATION_COUNT,
        RIVER_CLUSTER_COUNT,
        TURN_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Simulated Turn hand strengths in {:?}.", elapsed_time);

    println!("Clustering Flop combos.");
    start_time = Instant::now();
    let (centroids, clusters) = cluster::kmeans(
        &result,
        FLOP_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Clustered Flop combos in {:?}.", elapsed_time);

    println!("Writing Flop combos.");
    start_time = Instant::now();
    save_combos(&flop_combos, "./output/flop_combos.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Flop combos in {:?}.", elapsed_time);

    println!("Writing Flop centroids.");
    start_time = Instant::now();
    save_centroids(&centroids, "./output/flop_centroids.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Flop centroids in {:?}.", elapsed_time);
    
    println!("Writing Flop clusters.");
    start_time = Instant::now();
    save_clusters(&clusters, "./output/flop_clusters.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Flop clusters in {:?}.", elapsed_time);

    (centroids, clusters)
}


fn build_lut() {
    // let deck = combo::create_deck(2, 4);
    let deck = combo::create_deck(2, 14);
    let start_combos = combo::create_card_combos(&deck, 2);
    let lookup = card::load_lookup("./assets/lookup.json");

    let (river_centroids, river_clusters) = build_river_lut(
        &deck,
        &start_combos,
        &lookup,
    );
    let (turn_centroids, turn_clusters) = build_turn_lut(
        &deck,
        &start_combos,
        &lookup,
        &river_centroids,
    );
    let (flop_centroids, flop_clusters) = build_flop_lut(
        &deck,
        &start_combos,
        &lookup,
        &river_centroids,
        &turn_centroids,
    );
}


fn main() {
    // test::test();
    build_lut();
}
