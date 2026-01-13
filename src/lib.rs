pub mod core;
pub mod ops;

// defining the user-facing macro
#[macro_export]
macro_rules! matrix 
{
    ( $( [ $( $x:expr ),* $(,)? ] ),* $(,)? ) => 
    {
        {
            let mut data = Vec::new();
            let mut rows = 0;
            let mut cols = 0;

            // outer loop
            $(
                rows += 1;
                let mut current_row_cols = 0;

                // inner loop
                $(
                    data.push($x);
                    current_row_cols += 1;
                )*

                // verify rectangular matrix
                if rows == 1
                {
                    cols = current_row_cols;
                }
                else if current_row_cols != cols
                {
                    panic!("matrix! format error: Row {} has {} cols, expected {}", rows, current_row_cols, cols);
                }
            )*

            $crate::core::matrix::Matrix::new(rows, cols, data).expect("Failed to create matrix via macro")
        }
    };
}