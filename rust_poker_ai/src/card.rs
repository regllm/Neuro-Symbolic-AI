use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use itertools::Itertools;


pub struct LookupTable {
    pub flush: HashMap<i32, i32>,
    pub unsuited: HashMap<i32, i32>,
}

fn get_int_rank(raw_rank: u8) -> i32 {
    (raw_rank as i32) - 2
}

fn get_int_suit(suit: char) -> i32 {
    match suit {
        's' => 1,
        'h' => 2,
        'd' => 4,
        'c' => 8,
        _ => -1,
    }
}

fn get_prime_rank(rank_int: i32) -> i32 {
    match rank_int {
        0 => 2,
        1 => 3,
        2 => 5,
        3 => 7,
        4 => 11,
        5 => 13,
        6 => 17,
        7 => 19,
        8 => 23,
        9 => 29,
        10 => 31,
        11 => 37,
        12 => 41,
        _ => -1,
    }
}

fn prime_product_from_rankbits(rankbits: i32) -> i32 {
    let mut product = 1;
    for i in 0..=12 {
        // if the ith bit is set
        if rankbits & (1 << i) != 0 {
            product *= get_prime_rank(i);
        }
    }
    product
}

fn prime_product_from_hand(card_ints: &Vec<&i32>) -> i32 {
    let mut product = 1;
    for &c in card_ints.iter() {
        product *= c & 0xFF;
    }
    product
}

pub fn new(raw_rank: u8, char_suit: char) -> i32 {
    let int_rank = get_int_rank(raw_rank);
    let int_suit = get_int_suit(char_suit);
    let prime_rank = get_prime_rank(int_rank);

    let bit_rank = 1 << (int_rank as i32) << 16;
    let suit = int_suit << 12;
    let rank = (int_rank as i32) << 8;

    bit_rank | suit | rank | prime_rank
}

pub fn get_rank_int(card_int: i32) -> i32 {
    (card_int >> 8) & 0xF
}

pub fn get_suit_int(card_int: i32) -> i32 {
    (card_int >> 12) & 0xF
}

pub fn get_small_suit_int(card_int: i32) -> i32 {
    match get_suit_int(card_int) {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => -1,
    }
}

pub fn get_card_key(card_int: i32) -> i32 {
    get_rank_int(card_int) * 4 + get_small_suit_int(card_int)
}

pub fn get_combo_key(card_ints: &Vec<i32>) -> u64 {
    let mut result: u64 = 0;
    for (i, &card_int) in card_ints.iter().enumerate() {
        result += (get_card_key(card_int) + 1) as u64 * u64::pow(64, i as u32);
    }
    result
}

pub fn load_lookup(file_path: &str) -> LookupTable {
    let mut file = File::open(file_path).expect("Failed to open file");
    let mut text = String::new();
    file.read_to_string(&mut text).expect("Failed to read file");

    let raw_tables: HashMap<String, HashMap<String, Value>> = serde_json::from_str(&text).expect("Failed to parse JSON");

    let flush: HashMap<i32, i32> = raw_tables["flush"].clone().into_iter()
        .map(|(k, v)| (k.parse().expect("Failed to parse key"), v.as_i64().expect("Failed to parse value") as i32))
        .collect();

    let unsuited: HashMap<i32, i32> = raw_tables["unsuited"].clone().into_iter()
        .map(|(k, v)| (k.parse().expect("Failed to parse key"), v.as_i64().expect("Failed to parse value") as i32))
        .collect();

    LookupTable {
        flush: flush,
        unsuited: unsuited,
    }
}

pub fn evaluate_five_cards(cards: &Vec<&i32>, lookup: &LookupTable) -> i32 {
    if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000 != 0 {
        let hand_or = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16;
        let prime = prime_product_from_rankbits(hand_or);
        lookup.flush[&prime]
    } else {
        let prime = prime_product_from_hand(cards);
        lookup.unsuited[&prime]
    }
}

pub fn evaluate(cards: &Vec<i32>, lookup: &LookupTable) -> i32 {
    if cards.len() == 5 {
        evaluate_five_cards(&cards.iter().collect(), lookup)
    } else {
        let mut min_score = 7462;  // LookupTable.MAX_HIGH_CARD

        for combo in cards.iter().combinations(5) {
            let score = evaluate_five_cards(&combo, lookup);
            if score < min_score {
                min_score = score;
            }
        }

        min_score
    }
}
