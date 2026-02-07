mod bps_sada;
mod rank_support_v;
mod select_support_mcl;
mod traits;
mod util;

use std::sync::Arc;
use bitvec::prelude::*;

use crate::{bps_sada::BpSupportSadaBitvec, rank_support_v::RankSupportV, traits::{Pat1, Sel1}};

// ---- paste the whole module from the previous message above this line ----
// or put it in bp_support_sada.rs and `mod bp_support_sada; use bp_support_sada::*;`

fn bits_from_bp_string(s: &str) -> Vec<bool> {
    s.chars()
        .map(|c| match c {
            '(' => true,
            ')' => false,
            _ => panic!("invalid char {c:?}, expected '(' or ')'"),
        })
        .collect()
}

fn bitvec_from_bools(bits: &[bool]) -> BitVec<u64, Lsb0> {
    let mut bv: BitVec<u64, Lsb0> = BitVec::with_capacity(bits.len());
    bv.extend(bits.iter().copied());
    bv
}

fn main() {

    benchmark();
    return;

    // Example balanced parentheses sequence: "(()())()"
    // Indices:  0 1 2 3 4 5 6 7
    // Bits:     1 1 0 1 0 0 1 0
    let bp_str = "(()())()";
    let bits = bits_from_bp_string(bp_str);
    let bv = Arc::new(bitvec_from_bools(&bits));

    // Build Sadakane support structure
    // (Using default SML=256 and MED_DEG=32 and SimpleRank/SimpleSelect backends.)
    let bp = BpSupportSadaBitvec::<256, 32>::build(bv);

    println!("BP string: {bp_str}");
    println!("Length: {}", bp.len());
    println!();

    // Show excess array
    println!("excess(i) for i=0..n-1:");
    for i in 0..bp.len() {
        print!("{:2}: {:2}   ", i, bp.excess(i));
    }
    println!("\n");

    // rank(i) = #opens up to and including i
    println!("rank(i) (#opens up to i):");
    for i in 0..bp.len() {
        print!("{:2}: {:2}   ", i, bp.rank(i));
    }
    println!("\n");

    // select(k): position of k-th open (k>=1)
    let total_opens = bp.rank(bp.len() - 1);
    println!("select(k) positions of opens:");
    for k in 1..=total_opens {
        println!("  select({k}) = {}", bp.select(k));
    }
    println!();

    // Demonstrate find_close for each open
    println!("find_close(i) for each '(' position:");
    for i in 0..bp.len() {
        // we don't have direct access to the bitvector here,
        // but for demo we can re-parse bp_str:
        if bp_str.as_bytes()[i] == b'(' {
            let j = bp.find_close(i);
            println!("  open at {i} closes at {j}   substring: {}", &bp_str[i..=j]);
        }
    }
    println!();

    // Demonstrate find_open for each close
    println!("find_open(i) for each ')' position:");
    for i in 0..bp.len() {
        if bp_str.as_bytes()[i] == b')' {
            let j = bp.find_open(i);
            println!("  close at {i} opens at {j}   substring: {}", &bp_str[j..=i]);
        }
    }
    println!();

    // Demonstrate enclose for each position
    println!("enclose(i): nearest enclosing '(' for i (or n if none):");
    for i in 0..bp.len() {
        let e = bp.enclose(i);
        if e < bp.len() {
            let c = bp.find_close(e);
            println!("  i={i}: enclose={e}, pair=({e},{c}) -> {}", &bp_str[e..=c]);
        } else {
            println!("  i={i}: enclose = n (none)");
        }
    }
}

fn benchmark() {
    // Generate a bitvector of with 8 billion bits randomly set
    let n = 8_000_000_000usize;
    let mut bv: BitVec<u64, Lsb0> = BitVec::with_capacity(n);
    println!("Generating random bitvector of length {n}...");
    for _ in 0..n {
        bv.push(rand::random::<bool>());
    }
    let n_ones = bv.count_ones();

    print!("Bitvector generated. Building RankSupportV... ");
    let bv = Arc::new(bv);
    let rank_support = RankSupportV::<Pat1>::new(bv.clone());

    // Draw 1 million random query positions
    let n_queries = 1_000_000;
    println!("Generating {n_queries} random query positions...");
    let mut query_positions: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let pos = rand::random::<u64>() as usize % n;
        query_positions.push(pos);
    }

    println!("Running rank queries on RankSupportV...");

    // Start the timer
    let start = std::time::Instant::now();

    // Run the queries
    let mut sum_of_answers = 0_usize;
    for &pos in &query_positions {
        sum_of_answers += rank_support.rank(pos);
    }
    println!("Sum of all rank answers: {}", sum_of_answers);

    // Print the elapsed time per query in nanoseconds
    let elapsed = start.elapsed();
    let avg_time_per_query = elapsed.as_secs_f64() / n_queries as f64;
    println!("Average time per rank query: {:.2} ns", avg_time_per_query * 1e9);

    // Benchmark select queries
    let select_support = crate::select_support_mcl::SelectSupportMcl::<Sel1>::new(bv.clone());
    let max_select_query = n_ones;
    let mut select_query_positions: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let pos = rand::random::<u64>() as usize % max_select_query;
        select_query_positions.push(pos);
    }

    println!("Running select queries on SelectSupportMcl...");

    // Start the timer
    let start = std::time::Instant::now();

    // Run the queries
    let mut sum_of_select_answers = 0_usize;
    for &pos in &select_query_positions {
        sum_of_select_answers += select_support.select(pos);
    }
    println!("Sum of all select answers: {}", sum_of_select_answers);

    // Print the elapsed time per query in nanoseconds
    let elapsed = start.elapsed();
    let avg_time_per_select_query = elapsed.as_secs_f64() / n_queries as f64;
    println!("Average time per select query: {:.2} ns", avg_time_per_select_query * 1e9);

}