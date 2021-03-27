import cv2

def AnnotateSudoku(cell_loactions, board, sudoku_original):
    for row_index, row in enumerate(cell_locations):
        for col_index, cell in enumerate(row):
            cv2.putText(sudoku_original, 
                        str(board[row_index][col_index]), 
                        (cell[0]+5, cell[3]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0,0,0), 
                        1, 
                        cv2.LINE_AA)
    return sudoku_original
