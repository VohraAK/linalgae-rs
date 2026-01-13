use linalgae_rs::{matrix, core::matrix::Matrix};

// -------- Basic Macro Tests --------

#[test]
fn test_macro_single_element() {
    let m = matrix![[42]];
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 1);
    assert_eq!(m.as_slice()[0], 42);
}

#[test]
fn test_macro_single_row() {
    let m = matrix![[1, 2, 3, 4]];
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 4);
    assert_eq!(m.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn test_macro_single_column() {
    let m = matrix![[1], [2], [3], [4]];
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 1);
    assert_eq!(m.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn test_macro_square_matrix() {
    let m = matrix![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ];
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_macro_rectangular_matrix() {
    let m = matrix![
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ];
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 4);
    assert_eq!(m.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

// -------- Type Inference Tests --------

#[test]
fn test_macro_f64_matrix() {
    let m = matrix![
        [1.0, 2.5],
        [3.7, 4.2]
    ];
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.as_slice(), &[1.0, 2.5, 3.7, 4.2]);
}

#[test]
fn test_macro_f32_matrix() {
    let m = matrix![
        [1.0_f32, 2.5_f32],
        [3.7_f32, 4.2_f32]
    ];
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.as_slice(), &[1.0_f32, 2.5_f32, 3.7_f32, 4.2_f32]);
}

#[test]
fn test_macro_i32_matrix() {
    let m = matrix![
        [1_i32, 2_i32],
        [3_i32, 4_i32]
    ];
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.as_slice(), &[1_i32, 2_i32, 3_i32, 4_i32]);
}

// -------- Expression Tests --------

#[test]
fn test_macro_with_expressions() {
    let x = 5;
    let y = 10;
    let m = matrix![
        [x + 1, y - 2],
        [x * 2, y / 2]
    ];
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.as_slice(), &[6, 8, 10, 5]);
}

#[test]
fn test_macro_with_function_calls() {
    fn double(x: i32) -> i32 { x * 2 }
    let m = matrix![
        [double(1), double(2)],
        [double(3), double(4)]
    ];
    assert_eq!(m.as_slice(), &[2, 4, 6, 8]);
}

#[test]
fn test_macro_with_constants() {
    const A: i32 = 100;
    const B: i32 = 200;
    let m = matrix![[A, B]];
    assert_eq!(m.as_slice(), &[100, 200]);
}

// -------- Special Value Tests --------

#[test]
fn test_macro_with_negative_numbers() {
    let m = matrix![
        [-1, -2],
        [3, -4]
    ];
    assert_eq!(m.as_slice(), &[-1, -2, 3, -4]);
}

#[test]
fn test_macro_with_zero() {
    let m = matrix![
        [0, 1],
        [0, 0]
    ];
    assert_eq!(m.as_slice(), &[0, 1, 0, 0]);
}

#[test]
fn test_macro_with_floats() {
    let m = matrix![
        [3.14159_f64, 2.71828_f64],
        [-1.41421_f64, 1.61803_f64]
    ];
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert!((m.as_slice()[0] - 3.14159_f64).abs() < 1e-10);
    assert!((m.as_slice()[1] - 2.71828_f64).abs() < 1e-10);
}

#[test]
fn test_macro_with_infinity() {
    let m = matrix![[f64::INFINITY, f64::NEG_INFINITY]];
    assert_eq!(m.as_slice()[0], f64::INFINITY);
    assert_eq!(m.as_slice()[1], f64::NEG_INFINITY);
}

#[test]
fn test_macro_with_nan() {
    let m = matrix![[f64::NAN, 1.0]];
    assert!(m.as_slice()[0].is_nan());
    assert_eq!(m.as_slice()[1], 1.0);
}

// -------- Integration Tests --------

#[test]
fn test_macro_with_arithmetic_operations() {
    let a = matrix![
        [1, 2],
        [3, 4]
    ];
    let b = matrix![
        [5, 6],
        [7, 8]
    ];
    let c = &a + &b;
    assert_eq!(c.as_slice(), &[6, 8, 10, 12]);
}

#[test]
fn test_macro_with_matrix_multiplication() {
    let a = matrix![
        [1, 2],
        [3, 4]
    ];
    let b = matrix![
        [5, 6],
        [7, 8]
    ];
    let c = &a * &b;
    assert_eq!(c.as_slice(), &[19, 22, 43, 50]);
}

#[test]
fn test_macro_identity_pattern() {
    let identity = matrix![
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ];
    let generated_identity = Matrix::<i32>::identity(3).unwrap();
    assert_eq!(identity.as_slice(), generated_identity.as_slice());
}

#[test]
fn test_macro_transpose() {
    let m = matrix![
        [1, 2, 3],
        [4, 5, 6]
    ];
    let t = m.t();
    let expected = matrix![
        [1, 4],
        [2, 5],
        [3, 6]
    ];
    assert_eq!(t.as_slice(), expected.as_slice());
}

#[test]
fn test_macro_scalar_operations() {
    let m = matrix![
        [1, 2],
        [3, 4]
    ];
    let doubled = &m * 2;
    let expected = matrix![
        [2, 4],
        [6, 8]
    ];
    assert_eq!(doubled.as_slice(), expected.as_slice());
}

// -------- Error Handling Tests --------

#[test]
#[should_panic(expected = "Row 2 has 3 cols, expected 2")]
fn test_macro_inconsistent_row_lengths() {
    let _ = matrix![
        [1, 2],
        [3, 4, 5]  // This row has 3 elements instead of 2
    ];
}

#[test]
#[should_panic(expected = "Row 3 has 1 cols, expected 3")]
fn test_macro_inconsistent_middle_row() {
    let _ = matrix![
        [1, 2, 3],
        [4, 5, 6],
        [7]        // This row has only 1 element instead of 3
    ];
}

// -------- Large Matrix Tests --------

#[test]
fn test_macro_large_matrix() {
    let m = matrix![
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
    ];
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 5);
    assert_eq!(m.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
}

// -------- Edge Cases --------

#[test]
fn test_macro_trailing_commas() {
    let m = matrix![
        [1, 2,],
        [3, 4,],
    ];
    assert_eq!(m.as_slice(), &[1, 2, 3, 4]);
}

#[test]
fn test_macro_single_trailing_comma() {
    let m = matrix![[1, 2, 3,]];
    assert_eq!(m.as_slice(), &[1, 2, 3]);
}

#[test]
fn test_macro_outer_trailing_comma() {
    let m = matrix![
        [1, 2],
        [3, 4],
    ];
    assert_eq!(m.as_slice(), &[1, 2, 3, 4]);
}

// -------- Complex Expression Tests --------

#[test]
fn test_macro_with_complex_expressions() {
    let base: i32 = 2;
    let m = matrix![
        [base.pow(0), base.pow(1), base.pow(2)],
        [base.pow(3), base.pow(4), base.pow(5)]
    ];
    assert_eq!(m.as_slice(), &[1, 2, 4, 8, 16, 32]);
}

#[test]
fn test_macro_with_variable_references() {
    let values = [10, 20, 30, 40];
    let m = matrix![
        [values[0], values[1]],
        [values[2], values[3]]
    ];
    assert_eq!(m.as_slice(), &[10, 20, 30, 40]);
}