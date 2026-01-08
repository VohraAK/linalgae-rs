use std::process;
use linalgae_rs::core::matrix::Matrix;

fn main()
{
    let matrix_1 = Matrix::zeroes(15, 12).unwrap_or_else(|e| 
        {
            eprintln!("Error: {e}");
            process::exit(1);
        });

    // println!("{}", matrix_1);
    // println!("Dim: {} x {}", matrix_1.rows(), matrix_1.cols());

    let matrix_2 = Matrix::new(4, 2, vec![2., 3., 4., 81., -3., 7.44, -1110., 29229.]).unwrap_or_else(|e| 
        {
            eprintln!("Error: {e}");
            process::exit(1);
    });

    let matrix_3 = Matrix::new(4, 2, vec![2., 3., 4., 81., -3., 7.44, -1110., 29229.]).unwrap_or_else(|e| 
        {
            eprintln!("Error: {e}");
            process::exit(1);
    });

    // let view = matrix_2.as_slice();
    // println!("\nPrinting elements...");

    // for &ele in view { print!("{}, ", ele);}

    let result = &matrix_2 + &matrix_3;
    println!("{}", result);

    let result = &matrix_1 - &matrix_3;     // panics due to dim mismatch
    println!("{}", result);


}