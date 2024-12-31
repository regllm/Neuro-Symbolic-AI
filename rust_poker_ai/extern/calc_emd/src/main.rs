extern crate emd;

use emd::vectors_distance;
use std::io::{self, BufRead};
use std::f64;

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while let Some(Ok(line)) = lines.next() {
        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() != 2 {
            eprintln!("Input should contain exactly one semicolon.");
            continue;
        }

        let vec1 = parse_vector(parts[0]);
        let vec2 = parse_vector(parts[1]);

        match (vec1, vec2) {
            (Some(v1), Some(v2)) => {
                match distance(&v1, &v2) {
                    Some(dist) => println!("{}", dist),
                    None => eprintln!("Vectors must be of the same length."),
                }
            },
            _ => eprintln!("Error parsing vectors."),
        }
    }
}

fn parse_vector(input: &str) -> Option<Vec<f64>> {
    let nums: Result<Vec<f64>, _> = input.split(',').map(str::parse).collect();
    nums.ok()
}

fn distance(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Option<f64> {
    Some(vectors_distance(vec1, vec2))
}