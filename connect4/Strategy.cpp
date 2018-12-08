#include <iostream>
#include "Point.h"
#include "Strategy.h"

const int MAX_INF = 1000000;
const int MIN_INF = -1000000;
const int deep = 5;//搜索树的深度
const int depth = 1;//价值的深度

using namespace std;


/*
	策略函数接口,该函数被对抗平台调用,每次传入当前状态,要求输出你的落子点,该落子点必须是一个符合游戏规则的落子点,不然对抗平台会直接认为你的程序有误

	input:
		为了防止对对抗平台维护的数据造成更改，所有传入的参数均为const属性
		M, N : 棋盘大小 M - 行数 N - 列数 均从0开始计， 左上角为坐标原点，行用x标记，列用y标记
		top : 当前棋盘每一列列顶的实际位置. e.g. 第i列为空,则_top[i] == M, 第i列已满,则_top[i] == 0
		_board : 棋盘的一维数组表示, 为了方便使用，在该函数刚开始处，我们已经将其转化为了二维数组board
				你只需直接使用board即可，左上角为坐标原点，数组从[0][0]开始计(不是[1][1])
				board[x][y]表示第x行、第y列的点(从0开始计)
				board[x][y] == 0/1/2 分别对应(x,y)处 无落子/有用户的子/有程序的子,不可落子点处的值也为0
		lastX, lastY : 对方上一次落子的位置, 你可能不需要该参数，也可能需要的不仅仅是对方一步的
				落子位置，这时你可以在自己的程序中记录对方连续多步的落子位置，这完全取决于你自己的策略
		noX, noY : 棋盘上的不可落子点(注:其实这里给出的top已经替你处理了不可落子点，也就是说如果某一步
				所落的子的上面恰是不可落子点，那么UI工程中的代码就已经将该列的top值又进行了一次减一操作，
				所以在你的代码中也可以根本不使用noX和noY这两个参数，完全认为top数组就是当前每列的顶部即可,
				当然如果你想使用lastX,lastY参数，有可能就要同时考虑noX和noY了)
		以上参数实际上包含了当前状态(M N _top _board)以及历史信息(lastX lastY),你要做的就是在这些信息下给出尽可能明智的落子点
	output:
		你的落子点Point
*/
extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board,
	const int lastX, const int lastY, const int noX, const int noY) {
	/*
		不要更改这段代码
	*/
	int x = -1, y = -1;//最终将你的落子点存到x,y中
	int** board = new int*[M];
	for (int i = 0; i < M; i++) {
		board[i] = new int[N];
		for (int j = 0; j < N; j++) {
			board[i][j] = _board[i * N + j];
		}
	}

	/*
		根据你自己的策略来返回落子点,也就是根据你的策略完成对x,y的赋值
		该部分对参数使用没有限制，为了方便实现，你可以定义自己新的类、.h文件、.cpp文件
	*/
	//Add your own code below

	//freopen("column.txt", "a", stdout); //输出重定向，输出数据将保存在文件中

	int * place_cursor = new int[N];
	for (int i = 0; i < N; i++)
		place_cursor[i] = top[i];

	int ** value = new int *[depth];
	for (int i = 0; i < depth; i++)
		value[i] = new int[N];

	for (int y = 0; y < N; y++)
	{
		if (top[y] == 0)//如果该列已满
			value[0][y] = 2 * MIN_INF;
		else
			value[0][y] = search(board, place_cursor, M, N, noX, noY, deep, 2, 1, y, MIN_INF, MAX_INF);
	}

	//cout << "value = " << endl;
	//for (int x = 0; x < depth; x++)
	//{
	//	for (int y = 0; y < N; y++)
	//	{
	//		cout << value[x][y] << " ";
	//	}
	//	cout << endl;
	//}

	int select = 0;
	for (int x = 0; x < depth; x++)
	{
		for (int y = 0; y < N; y++)
		{
			if (value[x][y] > value[x][select])
				select = y;
		}
		if (value[x][select] == MAX_INF)
			break;
	}
	x = place_cursor[select] - 1;
	y = select;

	//cout << "x=" << x << "   y=" <<y<< endl;

	for (int i = 0; i < depth; i++)
		delete[] value[i];
	delete[] value;
	delete[] place_cursor;

	/*
		不要更改这段代码
	*/
	clearArray(M, N, board);
	return new Point(x, y);
}


/*
	getPoint函数返回的Point指针是在本dll模块中声明的，为避免产生堆错误，应在外部调用本dll中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C" __declspec(dllexport) void clearPoint(Point* p) {
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int** board) {
	for (int i = 0; i < M; i++) {
		delete[] board[i];
	}
	delete[] board;
}


/*
	添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h .cpp文件来辅助实现你的想法
*/

int search(int** board, int * cursor, const int M, const int N, const int noX, const int noY, int depth, int player, int opponent, int column, int alpha, int beta)
{
	cursor[column]--;
	board[cursor[column]][column] = player;
	if (player == 2)
	{
		if (machineWin(cursor[column], column, M, N, board))//若我方AI赢得棋局，给予相当大的价值
		{
			board[cursor[column]][column] = 0;
			cursor[column]++;
			return MAX_INF;
		}
	}
	else //player == 1
	{
		if (userWin(cursor[column], column, M, N, board))//若输掉棋局，给予相当小的价值
		{
			board[cursor[column]][column] = 0;
			cursor[column]++;
			return MIN_INF;
		}
	}

	if (column == noY && cursor[column] - 1 == noX)//回避不可落子点
		cursor[column]--;

	if (depth <= 0)//到达搜索树叶子节点价值
	{
		int leaf = evaluate(M, N, board, noX, noY, cursor, 2) - evaluate(M, N, board, noX, noY, cursor, 1);

		if (column == noY && cursor[column] == noX)//游标归位时，回避不可落子点额外上移的一步要进行回退
			cursor[column]++;

		board[cursor[column]][column] = 0;
		cursor[column]++;
		return leaf;
	}

	if (player == 1)
	{
		for (int y = 0;y < N; y++)
			if (cursor[y] != 0)
			{
				int node = search(board, cursor, M, N, noX, noY, depth - 1, opponent, player, y, alpha, beta);
				if (alpha < node)
					alpha = node;
				if (beta <= alpha)
					break;
			}

		if (column == noY && cursor[column] == noX)//游标归位时，回避不可落子点额外上移的一步要进行回退
			cursor[column]++;

		board[cursor[column]][column] = 0;
		cursor[column]++;
		return alpha;
	}
	else //player == 2
	{
		for (int y = 0; y < N; y++)
			if (cursor[y] != 0)
			{
				int node = search(board, cursor, M, N, noX, noY, depth - 1, opponent, player, y, alpha, beta);
				if (beta > node)
					beta = node;
				if (beta <= alpha)
					break;
			}

		if (column == noY && cursor[column] == noX)//游标归位时，回避不可落子点额外上移的一步要进行回退
			cursor[column]++;

		board[cursor[column]][column] = 0;
		cursor[column]++;
		return beta;
	}
}

int evaluate(const int M, const int N, int** board, int noX, int noY, int * t, int player)
{
	//freopen("evaluate.txt", "a", stdout); //输出重定向，输出数据将保存在文件中
	int begin_x, begin_y, end_y, end_x;
	int count;//棋子计数值
	int interspace;//分段空白计数值
	int overall = 0;//总估价值
	int weight[4] = { 0,1,4,0 };

	//cout << "board = " << endl;
	//for (int x = 0; x < M; x++)
	//{
	//	for (int y = 0; y < N; y++)
	//	{
	//		cout << board[x][y] << " ";
	//	}
	//	cout << endl;
	//}
	//cout << "---------------------" << endl;
	//cout << "value table = " << endl;

	int gap_x, gap_y;//用于记录间断棋子
	int i, j, k;
	for (int x = 0; x < M; x++)
	{
		for (int y = 0; y < N; y++)
		{
			int value = 0;//单点价值
			if (board[x][y] == player)
			{
				//横向检测
				begin_y = y - 3;
				end_y = y + 3;
				while (begin_y < 0)
					begin_y++;
				while (end_y >= N)
					end_y--;

				for (i = begin_y; i <= end_y - 3; i++)
				{
					count = 0;
					interspace = 0;
					for (j = 0; j < 4; j++)
					{

						if (board[x][i + j] == player)
							count++;
						else
						{
							gap_y = i + j;
							if (board[x][i + j] == 0)
								interspace++;
						}

					}
					value += horizontal(board, M, N, t, player, noX, noY, x, i + j - 1, gap_y, interspace, count);
				}

				//纵向检测
				begin_x = x + 3;
				end_x = x - 3;
				while (begin_x >= M)
					begin_x--;
				while (end_x < 0)
					end_x++;

				for (i = begin_x; i >= end_x + 3; i--)
				{
					count = 0;
					for (j = 0; j < 4; j++)
						if (board[i - j][y] == player)
							count++;
					if (count == 3 && i - j + 1 >= 0 && !(i - j + 1 == noX && y == noY) && board[i - j + 1][y] == 0)
						value += 96;
					else
						value += weight[count];
				}

				//主对角线方向搜索
				begin_x = x - 3;
				begin_y = y - 3;
				end_x = x + 3;
				end_y = y + 3;
				while (begin_x < 0 || begin_y < 0)
				{
					begin_x++;
					begin_y++;
				}
				while (end_x >= M || end_y >= N)
				{
					end_x--;
					end_y--;
				}

				for (i = begin_x, j = begin_y; i <= end_x - 3, j <= end_y - 3; i++, j++)
				{
					count = 0;
					interspace = 0;
					for (k = 0; k < 4; k++)
					{
						if (board[i + k][j + k] == player)
							count++;
						else
						{
							gap_x = i + k;
							gap_y = j + k;
							if (board[i + k][j + k] == 0)
								interspace++;
						}
					}
					value += main_diagonal(board, M, N, t, player, noX, noY, i + k - 1, j + k - 1, gap_x, gap_y, interspace, count);
				}

				//次对角线方向搜索
				begin_x = x - 3;
				begin_y = y + 3;
				end_x = x + 3;
				end_y = y - 3;
				while (begin_x < 0 || begin_y >= N)
				{
					begin_x++;
					begin_y--;
				}
				while (end_x >= M || end_y < 0)
				{
					end_x--;
					end_y++;
				}
				for (i = begin_x, j = begin_y; i <= end_x - 3, j >= end_y + 3; i++, j--)
				{
					count = 0;
					interspace = 0;
					for (k = 0; k < 4; k++)
					{
						if (board[i + k][j - k] == player)
							count++;
						else
						{
							gap_x = i + k;
							gap_y = j - k;
							if (board[i + k][j - k] == 0)
								interspace++;
						}
					}
					value += secondary_diagonal(board, M, N, t, player, noX, noY, i + k - 1, j - k + 1, gap_x, gap_y, interspace, count);
				}
			}
			//cout << value << " ";
			overall += value;
		}
		//cout << endl;
	}
	//cout << "---------------------" << endl;
	return overall;
}


int horizontal(int ** board, const int M, const int N, int * t, int player, int noX, int noY, int x, int y, int gap_y, int interspace, int count)
{
	int except = 0;
	int value = 0;
	int weight[4] = { 0,1,4,0 };

	if (count == 3)
	{
		if (gap_y == y - 3)//若非本方棋子是四个中最左边的
		{
			if (!(noX == x && noY == y - 3) && board[x][y - 3] == 0)
				except++;
			if (y + 1 < N && !(noX == x && noY == y + 1) && board[x][y + 1] == 0)
				except++;

		}
		else if (gap_y == y)//若非本方棋子是四个中最右边的
		{
			if (!(noX == x && noY == y) && board[x][y] == 0)
				except++;
			if (y - 4 >= 0 && !(noX == x && noY == y - 4) && board[x][y - 4] == 0)
				except++;
		}
		else
		{
			if (!(noX == x && noY == gap_y) && board[x][gap_y] == 0)
				except += 2;
		}
		if (except == 1)
			return 48;
		else if (except == 2)
			return 96;
	}
	else if (count == 2)
	{
		if (interspace == 2 && y - 4 >= 0 && !(noX == x && noY == y - 4) && board[x][y - 4] == 0)
			return 48;
	}
	else
		value += weight[count];

	return value;
}

int main_diagonal(int ** board, const int M, const int N, int * t, int player, int noX, int noY, int x, int y, int gap_x, int gap_y, int interspace, int count)
{
	int except = 0;
	int value = 0;
	int weight[4] = { 0,1,4,0 };


	if (count == 3)
	{
		if (gap_x == x - 3 && gap_y == y - 3)//若非本方棋子是左侧的
		{
			if (!(noX == x - 3 && noY == y - 3) && board[x - 3][y - 3] == 0)
				except++;
			if (x + 1 < M && y + 1 < N && !(noX == x + 1 && noY == y + 1) && board[x + 1][y + 1] == 0)
				except++;
		}
		else if (gap_x == x && gap_y == y)//若非本方棋子是右侧的
		{
			if (!(noX == x && noY == y) && board[x][y] == 0)
				except++;
			if (x - 4 >= 0 && y - 4 >= 0 && !(noX == x - 4 && noY == y - 4) && board[x - 4][y - 4] == 0)
				except++;
		}
		else
		{
			if (!(noX == gap_x && noY == gap_y) && board[gap_x][gap_y] == 0)
				except += 2;
		}

		if (except == 1)
			return 48;
		else if (except == 2)
			return 96;
	}
	else if (count == 2)
	{
		if (interspace == 2)
			return 24;
	}
	else if (count == 0 || count == 1 || count == 2)
		value += weight[count];

	return value;
}

int secondary_diagonal(int ** board, const int M, const int N, int * t, int player, int noX, int noY, int x, int y, int gap_x, int gap_y, int interspace, int count)
{
	int except = 0;
	int value = 0;
	int weight[4] = { 0,1,4,0 };


	if (count == 3)
	{
		if (gap_x == x - 3 && gap_y == y + 3)//若非本方棋子是顶部的
		{
			if (!(noX == x - 3 && noY == y + 3) && board[x - 3][y + 3] == 0)
				except++;
			if (x + 1 < M && y - 1 >= 0 && !(noX == x + 1 && noY == y - 1) && board[x + 1][y - 1] == 0)
				except++;
		}
		else if (gap_x == x && gap_y == y)//若非本方棋子是底部的
		{
			if (!(noX == x && noY == y) && board[x][y] == 0)
				except++;
			if (x - 4 >= 0 && y + 4 < N && !(noX == x - 4 && noY == y + 4) && board[x - 4][y + 4] == 0)
				except++;
		}
		else
		{
			if (!(noX == gap_x && noY == gap_y) && board[gap_x][gap_y] == 0)
				except += 2;
		}

		if (except == 1)
			return 48;
		else if (except == 2)
			return 96;
	}
	else if (count == 2)
	{
		if (interspace == 2)
			return 24;
	}
	else if (count == 0 || count == 1 || count == 2)
		value += weight[count];

	return value;
}



