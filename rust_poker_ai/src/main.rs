mod eval;

fn main() {
    let result = eval::new(14, 's');
    println!("Card representation for Spade A: {}", result);
}