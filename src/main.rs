mod rank_support_v;
mod select_support_mcl;
mod traits;
mod util;

use std::sync::Arc;
use bitvec::prelude::*;
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::{RngCore, SeedableRng}};

use crate::{rank_support_v::RankSupportV, traits::{Pat1, Sel1}};
fn main() {

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let n = 8_000_000_000usize;
    let mut bv: BitVec<u64, Lsb0> = BitVec::with_capacity(n);
    println!("Generating random bitvector of length {n}...");
    for _ in 0..n {
        bv.push(rng.next_u64() % 2 == 0);
    }

    let bv = Arc::new(bv);
    benchmark_rank(bv.clone());
    benchmark_select(bv.clone());
}

fn benchmark_rank(bv: Arc<BitVec<u64, Lsb0>>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(131313);
    let n = bv.len();

    print!("Bitvector generated. Building RankSupportV... ");
    let rank_support = RankSupportV::<Pat1>::new(bv.clone());

    // Draw 1 million random query positions
    let n_queries = 1_000_000;
    println!("Generating {n_queries} random query positions...");
    let mut query_positions: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let pos = rng.next_u64() as usize % n;
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
}

fn benchmark_select(bv: Arc<BitVec<u64, Lsb0>>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(131313);
    let n_ones = bv.count_ones();

    // Benchmark select queries
    let n_queries = 1_000_000;
    let select_support = crate::select_support_mcl::SelectSupportMcl::<Sel1>::new(bv.clone());
    let max_select_query = n_ones;
    let mut select_query_positions: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let pos = rng.next_u64() as usize % max_select_query;
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