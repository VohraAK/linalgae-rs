use linalgae_rs::core::matrix::Matrix;

fn main()
{
    // TODO: make macros for Matrix init
    
    // let _matrix_1 = Matrix::<f64>::zeroes(15, 12).unwrap_or_else(|e| 
    //     {
    //         eprintln!("Error: {e}");
    //         process::exit(1);
    //     });

    // // println!("{}", matrix_1);
    // // println!("Dim: {} x {}", matrix_1.rows(), matrix_1.cols());

    // let _matrix_2 = Matrix::<f64>::new(4, 2, vec![2., 3., 4., 81., -3., 7.44, -1110., 29229.]).unwrap_or_else(|e| 
    //     {
    //         eprintln!("Error: {e}");
    //         process::exit(1);
    //     });

    // let _matrix_3 = Matrix::<f64>::new(4, 2, vec![2., 3., 4., 81., -3., 7.44, -1110., 29229.]).unwrap_or_else(|e| 
    //     {
    //         eprintln!("Error: {e}");
    //         process::exit(1);
    //     });

    // let _matrix_4 = Matrix::<i32>::identity(4).unwrap_or_else(|e| 
    //     {
    //         eprintln!("Error: {e}");
    //         process::exit(1);
    //     });

    // println!("{}", _matrix_4);

    // let view = matrix_2.as_slice();
    // println!("\nPrinting elements...");

    // for &ele in view { print!("{}, ", ele);}

    // let result = &matrix_2 + &matrix_3;
    // println!("{}", result);

    // let result = &matrix_1 - &matrix_3;     // panics due to dim mismatch
    // println!("{}", result);

    // TODO: shift ops to more generic types

    // let result_1 = (3 as f64) * &_matrix_2;
    // let result_2 = (&_matrix_2) / (3 as f64);

    // println!("result_1: {}", result_1);
    // println!("result_2: {}", result_2);

    // assert_eq!(result_1, result_2);

    // let matrix_1 = matrix![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]];

    // println!("\nmatrix_1: \n{}", matrix_1);

    // let matrix_2 = matrix_1.apply(|x| x * x);

    // println!("\nmatrix_2: \n{}", matrix_2);
    
    // let matrix_3 = matrix_2.t().apply(|y| y / (2.0*y + 1.0));

    // println!("\nmatrix_3: \n{}", matrix_3);

    let rand_1 = Matrix::<f64>::rand_init(5, 4);

    println!("{}", rand_1);
    
}