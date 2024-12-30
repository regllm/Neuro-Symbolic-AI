use crate::args;
use crate::cluster::{strength, util};
use crate::poker::{card, combo};
use crate::progress;

use std::time::{Instant};
use std::fs::{File, metadata};
use std::io::{Write, BufReader, BufRead};


const FLOP_SIMULATION_COUNT: u32 = 12;
const TURN_SIMULATION_COUNT: u32 = 12;
const RIVER_SIMULATION_COUNT: u32 = 12;
const FLOP_CLUSTER_COUNT: u32 = 50;
const TURN_CLUSTER_COUNT: u32 = 50;
const RIVER_CLUSTER_COUNT: u32 = 50;
const TURN_CLUSTER_COUNT_LIMIT: u32 = 50;
const RIVER_CLUSTER_COUNT_LIMIT: u32 = 50;


fn load_centroids(file_path: &str) -> Vec<Vec<f64>> {
    let mut file = File::open(file_path).unwrap();
    let mut reader = BufReader::new(file);
    let line_count = reader.lines().count();

    file = File::open(file_path).unwrap();
    reader = BufReader::new(file);

    let mut result: Vec<Vec<f64>> = Vec::with_capacity(line_count);

    let progress = progress::new(line_count as u64);

    for line in reader.lines() {
        let line = line.unwrap();
        let values: Vec<f64> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        result.push(values);
        progress.inc(1);
    }
    progress.finish();

    result
}

fn save_combos(combos: &Vec<Vec<i32>>, file_path: &str) {
    let progress = progress::new(combos.len() as u64);

    let mut file = File::create(file_path).unwrap();
    for row in combos.iter() {
        let row_str = row.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(",");
        writeln!(file, "{}", row_str).unwrap();
        progress.inc(1);
    }

    progress.finish();
}

fn save_centroids(centroids: &Vec<Vec<f64>>, file_path: &str) {
    let progress = progress::new(centroids.len() as u64);

    let mut file = File::create(file_path).unwrap();
    for row in centroids.iter() {
        let row_str = row.iter().map(|&x| x.to_string()).collect::<Vec<String>>().join(",");
        writeln!(file, "{}", row_str).unwrap();
        progress.inc(1);
    }

    progress.finish();
}

fn save_clusters(clusters: &Vec<u32>, file_path: &str) {
    let progress = progress::new(clusters.len() as u64);

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

    println!("Clustering River combos.");
    start_time = Instant::now();
    let (centroids, clusters) = util::kmeans(
        &result,
        RIVER_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Clustered River combos in {:?}.", elapsed_time);

    let limited_centroids;
    if RIVER_CLUSTER_COUNT > RIVER_CLUSTER_COUNT_LIMIT {
        println!("Clustering River combos with limited cluster counts.");
        start_time = Instant::now();
        let (small_centroids, _small_clusters) = util::kmeans(
            &result,
            RIVER_CLUSTER_COUNT_LIMIT,
        );
        elapsed_time = Instant::now() - start_time;
        println!("Clustered River combos in {:?}.", elapsed_time);
        limited_centroids = small_centroids;
    } else {
        limited_centroids = centroids.clone();
    }

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

    println!("Writing River centroids with limited cluster counts.");
    start_time = Instant::now();
    save_centroids(&limited_centroids, "./output/river_centroids_limited.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River centroids in {:?}.", elapsed_time);
    
    println!("Writing River clusters.");
    start_time = Instant::now();
    save_clusters(&clusters, "./output/river_clusters.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote River clusters in {:?}.", elapsed_time);

    (limited_centroids, clusters)
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
        RIVER_CLUSTER_COUNT_LIMIT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Simulated Turn hand strengths in {:?}.", elapsed_time);

    println!("Clustering Turn combos.");
    start_time = Instant::now();
    let (centroids, clusters) = util::kmeans(
        &result,
        TURN_CLUSTER_COUNT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Clustered Turn combos in {:?}.", elapsed_time);

    let limited_centroids;
    if TURN_CLUSTER_COUNT > TURN_CLUSTER_COUNT_LIMIT {
        println!("Clustering Turn combos with limited cluster counts.");
        start_time = Instant::now();
        let (small_centroids, _small_clusters) = util::kmeans(
            &result,
            TURN_CLUSTER_COUNT_LIMIT,
        );
        elapsed_time = Instant::now() - start_time;
        println!("Clustered Turn combos in {:?}.", elapsed_time);
        limited_centroids = small_centroids;
    } else {
        limited_centroids = centroids.clone();
    }

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

    println!("Writing Turn centroids with limited cluster counts.");
    start_time = Instant::now();
    save_centroids(&limited_centroids, "./output/turn_centroids_limited.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Turn centroids in {:?}.", elapsed_time);
    
    println!("Writing Turn clusters.");
    start_time = Instant::now();
    save_clusters(&clusters, "./output/turn_clusters.txt");
    elapsed_time = Instant::now() - start_time;
    println!("Wrote Turn clusters in {:?}.", elapsed_time);

    (limited_centroids, clusters)
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

    println!("Simulating Flop hand strengths.");
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
        RIVER_CLUSTER_COUNT_LIMIT,
        TURN_CLUSTER_COUNT_LIMIT,
    );
    elapsed_time = Instant::now() - start_time;
    println!("Simulated Flop hand strengths in {:?}.", elapsed_time);

    println!("Clustering Flop combos.");
    start_time = Instant::now();
    let (centroids, clusters) = util::kmeans(
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


pub fn build_lut() {
    let cluster_args = args::get_cluster_args().unwrap();

    let min_rank = if cluster_args.short { 10 } else { 2 };
    let max_rank = 14;
    let deck = combo::create_deck(min_rank, max_rank);
    let start_combos = combo::create_card_combos(&deck, 2);
    let lookup = card::load_lookup("./assets/lookup.json");

    let (river_centroids, turn_centroids);
    if metadata("./output/river_centroids_limited.txt").is_ok() {
        println!("Loading River centroids.");
        river_centroids = load_centroids("./output/river_centroids_limited.txt");
    } else {
        let (centroids, _river_clusters) = build_river_lut(
            &deck,
            &start_combos,
            &lookup,
        );
        river_centroids = centroids;
    }
    if metadata("./output/turn_centroids_limited.txt").is_ok() {
        println!("Loading Turn centroids.");
        turn_centroids = load_centroids("./output/turn_centroids_limited.txt");
    } else {
        let (centroids, _turn_clusters) = build_turn_lut(
            &deck,
            &start_combos,
            &lookup,
            &river_centroids,
        );
        turn_centroids = centroids;
    }
    if !metadata("./output/flop_centroids.txt").is_ok() {
        let (_flop_centroids, _flop_clusters) = build_flop_lut(
            &deck,
            &start_combos,
            &lookup,
            &river_centroids,
            &turn_centroids,
        );
    }
}