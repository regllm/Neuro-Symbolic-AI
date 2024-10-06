use crate::args;
use crate::poker::card;
use crate::progress;

use indicatif::{ProgressBar};
use std::collections::HashMap;
use std::fs::{File};
use std::io::{BufReader, BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};


struct ClusterLookupTable {
    pub flop: HashMap<u64, u8>,
    pub turn: HashMap<u64, u8>,
    pub river: HashMap<u64, u8>,
}

static mut CLUSTER_LOOKUP_TABLE: Option<ClusterLookupTable> = None;


fn load_lookup_table(
    combos_file_path: &str,
    clusters_file_path: &str,
    table: &mut HashMap<u64, u8>,
) {
    let mut clusters_file = File::open(clusters_file_path).unwrap();
    let mut clusters_reader = BufReader::new(clusters_file);
    let line_count = clusters_reader.lines().count();

    clusters_file = File::open(clusters_file_path).unwrap();
    clusters_reader = BufReader::new(clusters_file);

    let combos_file = File::open(combos_file_path).unwrap();
    let combos_reader = BufReader::new(combos_file);


    let progress = ProgressBar::new(line_count as u64);
    progress.set_style(progress::create_progress_style());

    for (cluster_line, combo_line) in clusters_reader.lines().zip(combos_reader.lines()) {
        let combo_line = combo_line.unwrap();
        let cards: Vec<i32> = combo_line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let combo_key = card::get_combo_key(&cards);

        let cluster_line = cluster_line.unwrap();
        let cluster = cluster_line.parse::<u8>().unwrap();

        table.insert(combo_key, cluster);

        progress.inc(1);
    }
    progress.finish();
}

fn load_lookup_tables(
    tables_path: &str,
) {
    let mut flop_lookup_table: HashMap<u64, u8> = HashMap::new();
    let mut turn_lookup_table: HashMap<u64, u8> = HashMap::new();
    let mut river_lookup_table: HashMap<u64, u8> = HashMap::new();

    println!("Loading Flop combos and clusters.");
    let mut combos_file_path = tables_path.to_string() + "/flop_combos.txt";
    let mut clusters_file_path = tables_path.to_string() + "/flop_clusters.txt";
    load_lookup_table(&combos_file_path, &clusters_file_path, &mut flop_lookup_table);

    println!("Loading Turn combos and clusters.");
    combos_file_path = tables_path.to_string() + "/turn_combos.txt";
    clusters_file_path = tables_path.to_string() + "/turn_clusters.txt";
    load_lookup_table(&combos_file_path, &clusters_file_path, &mut turn_lookup_table);

    println!("Loading River combos and clusters.");
    combos_file_path = tables_path.to_string() + "/river_combos.txt";
    clusters_file_path = tables_path.to_string() + "/river_clusters.txt";
    load_lookup_table(&combos_file_path, &clusters_file_path, &mut river_lookup_table);

    // It is okay to set CLUSTER_LOOKUP_TABLE value
    // since threading is not started at all.
    unsafe {
        CLUSTER_LOOKUP_TABLE = Some(ClusterLookupTable{
            flop: flop_lookup_table,
            turn: turn_lookup_table,
            river: river_lookup_table,
        });
    }
}

fn handle_client(mut stream: TcpStream) {
    // let cluster_lookup_table = &CLUSTER_LOOKUP_TABLE.unwrap();
    let mut cards: Vec<i32> = Vec::new();
    let mut buffer = [0; 4];

    loop {
        match stream.read(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected");
                break;
            }
            Ok(_) => {
                let value = i32::from_be_bytes(buffer);

                if value == -1 {
                    // Get the cluster and clear the cards.
                    let combo_key = card::get_combo_key(&cards);
                    let cluster: i32;
                    // We are NOT mutating CLUSTER_LOOKUP_TABLE inside.
                    unsafe {
                        if let Some(table) = CLUSTER_LOOKUP_TABLE.as_ref() {
                            cluster = match cards.len() {
                                5 => *table.flop.get(&combo_key).unwrap() as i32,
                                6 => *table.turn.get(&combo_key).unwrap() as i32,
                                7 => *table.river.get(&combo_key).unwrap() as i32,
                                _ => -1,
                            };
                        } else {
                            cluster = -1;
                        }
                    }
                    cards.clear();
                    let response = cluster.to_be_bytes();
                    stream.write_all(&response).unwrap();
                } else {
                    // Collect the card.
                    cards.push(value);
                }
            }
            Err(_) => {
                println!("Error reading from client");
                break;
            }
        }
    }
}

fn start_server() {
    let table_args = args::get_table_args().unwrap();
    let address = format!("{}:{}", table_args.host, table_args.port);

    let listener = TcpListener::bind(address.clone());
    match listener {
        Ok(listener) => {
            println!("Server listening on {:?}...", address);

            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        std::thread::spawn(|| {
                            handle_client(stream);
                        });
                    }
                    Err(e) => {
                        eprintln!("Error accepting connection: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to bind to address: {}", e);
            println!("Restarting server...");
            start_server(); // Restart the server
        }
    }
}

pub fn run_server() {
    let table_args = args::get_table_args().unwrap();
    load_lookup_tables(&table_args.input);
    start_server();
}
