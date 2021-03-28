import cv2
from CropSudoku import find_puzzle
from SolveSudoku import SolveSudoku
from Annotator import AnnotateSudoku
from ClassifyDigits import ClassifyDigits

image = cv2.imread("./sudoku_images/sudoku3.jpeg")
# crop sudoku from given image
sudoku_original, sudoku_gray = find_puzzle(image)
# get cell_locations and classified digits from sudoku board
cell_locations, board = ClassifyDigits(sudoku_gray)
# solve sudoku
SolveSudoku.SolveSudoku(board)
# annotate the solved sudoku to the original sudoku board
solved_sudoku = AnnotateSudoku(cell_locations, board, sudoku_original)
cv2.imwrite("./sudoku_images/solved.png", solved_sudoku)