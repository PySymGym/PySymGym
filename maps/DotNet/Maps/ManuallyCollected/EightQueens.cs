using System;

namespace BranchesAndLoops
{
    public class EightQueens
    {
        private readonly int[][] _board;
        private const int N = 8;

        public EightQueens(int[][] board)
        {
                this._board = board;
                if (board.Length != N)
                {
                    throw new ArgumentException("Board length is not 64");
                }
        }

        private bool IsSafe(int row, int col)
        {

            // check if there is a queen in the same row to the
            // left
            for (int x = 0; x < col; x++)
                if (_board[row][x] == 1)
                    return false;

            // check if there is a queen in the upper left
            // diagonal
            for (int x = row, y = col; x >= 0 && y >= 0;
                 x--, y--)
                if (_board[x][y] == 1)
                    return false;

            // check if there is a queen in the lower left
            // diagonal
            for (int x = row, y = col; x < N && y >= 0;
                 x++, y--)
                if (_board[x][y] == 1)
                    return false;

            // if there is no queen in any of the above
            // positions, then it is safe to place a queen
            return true;
        }

        public bool CheckSolution()
        {
            if (this._board.Length == 0)
            {
                Console.WriteLine("Return false");
                return false;
            }

            int numQueens = 0;
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    if (_board[i][j] == 1)
                    {
                        numQueens++;
                    }
                }
            }

            if (numQueens != N)
            {
                return false;
            }

            for (int row = 0; row < this._board.Length/2; row++)
            {
                for (int col = 0; col < this._board.Length/2; col++)
                {
                    if (!IsSafe(row, col))
                    {
                        Console.WriteLine("Return false");
                        return false;
                    }
                }
            }

            Console.WriteLine("Return true");
            return true;
        }

    }
}


