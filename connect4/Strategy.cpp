#include <iostream>
#include "Point.h"
#include "Strategy.h"

const int MAX_INF = 1000000;
const int MIN_INF = -1000000;
const int deep = 5;//�����������
const int depth = 1;//��ֵ�����

using namespace std;


/*
	���Ժ����ӿ�,�ú������Կ�ƽ̨����,ÿ�δ��뵱ǰ״̬,Ҫ�����������ӵ�,�����ӵ������һ��������Ϸ��������ӵ�,��Ȼ�Կ�ƽ̨��ֱ����Ϊ��ĳ�������

	input:
		Ϊ�˷�ֹ�ԶԿ�ƽ̨ά����������ɸ��ģ����д���Ĳ�����Ϊconst����
		M, N : ���̴�С M - ���� N - ���� ����0��ʼ�ƣ� ���Ͻ�Ϊ����ԭ�㣬����x��ǣ�����y���
		top : ��ǰ����ÿһ���ж���ʵ��λ��. e.g. ��i��Ϊ��,��_top[i] == M, ��i������,��_top[i] == 0
		_board : ���̵�һά�����ʾ, Ϊ�˷���ʹ�ã��ڸú����տ�ʼ���������Ѿ�����ת��Ϊ�˶�ά����board
				��ֻ��ֱ��ʹ��board���ɣ����Ͻ�Ϊ����ԭ�㣬�����[0][0]��ʼ��(����[1][1])
				board[x][y]��ʾ��x�С���y�еĵ�(��0��ʼ��)
				board[x][y] == 0/1/2 �ֱ��Ӧ(x,y)�� ������/���û�����/�г������,�������ӵ㴦��ֵҲΪ0
		lastX, lastY : �Է���һ�����ӵ�λ��, ����ܲ���Ҫ�ò�����Ҳ������Ҫ�Ĳ������ǶԷ�һ����
				����λ�ã���ʱ��������Լ��ĳ����м�¼�Է������ಽ������λ�ã�����ȫȡ�������Լ��Ĳ���
		noX, noY : �����ϵĲ������ӵ�(ע:��ʵ���������top�Ѿ����㴦���˲������ӵ㣬Ҳ����˵���ĳһ��
				������ӵ�����ǡ�ǲ������ӵ㣬��ôUI�����еĴ�����Ѿ������е�topֵ�ֽ�����һ�μ�һ������
				��������Ĵ�����Ҳ���Ը�����ʹ��noX��noY��������������ȫ��Ϊtop������ǵ�ǰÿ�еĶ�������,
				��Ȼ�������ʹ��lastX,lastY�������п��ܾ�Ҫͬʱ����noX��noY��)
		���ϲ���ʵ���ϰ����˵�ǰ״̬(M N _top _board)�Լ���ʷ��Ϣ(lastX lastY),��Ҫ���ľ�������Щ��Ϣ�¸������������ǵ����ӵ�
	output:
		������ӵ�Point
*/
extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board,
	const int lastX, const int lastY, const int noX, const int noY) {
	/*
		��Ҫ������δ���
	*/
	int x = -1, y = -1;//���ս�������ӵ�浽x,y��
	int** board = new int*[M];
	for (int i = 0; i < M; i++) {
		board[i] = new int[N];
		for (int j = 0; j < N; j++) {
			board[i][j] = _board[i * N + j];
		}
	}

	/*
		�������Լ��Ĳ������������ӵ�,Ҳ���Ǹ�����Ĳ�����ɶ�x,y�ĸ�ֵ
		�ò��ֶԲ���ʹ��û�����ƣ�Ϊ�˷���ʵ�֣�����Զ����Լ��µ��ࡢ.h�ļ���.cpp�ļ�
	*/
	//Add your own code below

	//freopen("column.txt", "a", stdout); //����ض���������ݽ��������ļ���

	int * place_cursor = new int[N];
	for (int i = 0; i < N; i++)
		place_cursor[i] = top[i];

	int ** value = new int *[depth];
	for (int i = 0; i < depth; i++)
		value[i] = new int[N];

	for (int y = 0; y < N; y++)
	{
		if (top[y] == 0)//�����������
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
		��Ҫ������δ���
	*/
	clearArray(M, N, board);
	return new Point(x, y);
}


/*
	getPoint�������ص�Pointָ�����ڱ�dllģ���������ģ�Ϊ��������Ѵ���Ӧ���ⲿ���ñ�dll�е�
	�������ͷſռ䣬����Ӧ�����ⲿֱ��delete
*/
extern "C" __declspec(dllexport) void clearPoint(Point* p) {
	delete p;
	return;
}

/*
	���top��board����
*/
void clearArray(int M, int N, int** board) {
	for (int i = 0; i < M; i++) {
		delete[] board[i];
	}
	delete[] board;
}


/*
	������Լ��ĸ�������������������Լ����ࡢ����������µ�.h .cpp�ļ�������ʵ������뷨
*/

int search(int** board, int * cursor, const int M, const int N, const int noX, const int noY, int depth, int player, int opponent, int column, int alpha, int beta)
{
	cursor[column]--;
	board[cursor[column]][column] = player;
	if (player == 2)
	{
		if (machineWin(cursor[column], column, M, N, board))//���ҷ�AIӮ����֣������൱��ļ�ֵ
		{
			board[cursor[column]][column] = 0;
			cursor[column]++;
			return MAX_INF;
		}
	}
	else //player == 1
	{
		if (userWin(cursor[column], column, M, N, board))//�������֣������൱С�ļ�ֵ
		{
			board[cursor[column]][column] = 0;
			cursor[column]++;
			return MIN_INF;
		}
	}

	if (column == noY && cursor[column] - 1 == noX)//�رܲ������ӵ�
		cursor[column]--;

	if (depth <= 0)//����������Ҷ�ӽڵ��ֵ
	{
		int leaf = evaluate(M, N, board, noX, noY, cursor, 2) - evaluate(M, N, board, noX, noY, cursor, 1);

		if (column == noY && cursor[column] == noX)//�α��λʱ���رܲ������ӵ�������Ƶ�һ��Ҫ���л���
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

		if (column == noY && cursor[column] == noX)//�α��λʱ���رܲ������ӵ�������Ƶ�һ��Ҫ���л���
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

		if (column == noY && cursor[column] == noX)//�α��λʱ���رܲ������ӵ�������Ƶ�һ��Ҫ���л���
			cursor[column]++;

		board[cursor[column]][column] = 0;
		cursor[column]++;
		return beta;
	}
}

int evaluate(const int M, const int N, int** board, int noX, int noY, int * t, int player)
{
	//freopen("evaluate.txt", "a", stdout); //����ض���������ݽ��������ļ���
	int begin_x, begin_y, end_y, end_x;
	int count;//���Ӽ���ֵ
	int interspace;//�ֶοհ׼���ֵ
	int overall = 0;//�ܹ���ֵ
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

	int gap_x, gap_y;//���ڼ�¼�������
	int i, j, k;
	for (int x = 0; x < M; x++)
	{
		for (int y = 0; y < N; y++)
		{
			int value = 0;//�����ֵ
			if (board[x][y] == player)
			{
				//������
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

				//������
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

				//���Խ��߷�������
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

				//�ζԽ��߷�������
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
		if (gap_y == y - 3)//���Ǳ����������ĸ�������ߵ�
		{
			if (!(noX == x && noY == y - 3) && board[x][y - 3] == 0)
				except++;
			if (y + 1 < N && !(noX == x && noY == y + 1) && board[x][y + 1] == 0)
				except++;

		}
		else if (gap_y == y)//���Ǳ����������ĸ������ұߵ�
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
		if (gap_x == x - 3 && gap_y == y - 3)//���Ǳ�������������
		{
			if (!(noX == x - 3 && noY == y - 3) && board[x - 3][y - 3] == 0)
				except++;
			if (x + 1 < M && y + 1 < N && !(noX == x + 1 && noY == y + 1) && board[x + 1][y + 1] == 0)
				except++;
		}
		else if (gap_x == x && gap_y == y)//���Ǳ����������Ҳ��
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
		if (gap_x == x - 3 && gap_y == y + 3)//���Ǳ��������Ƕ�����
		{
			if (!(noX == x - 3 && noY == y + 3) && board[x - 3][y + 3] == 0)
				except++;
			if (x + 1 < M && y - 1 >= 0 && !(noX == x + 1 && noY == y - 1) && board[x + 1][y - 1] == 0)
				except++;
		}
		else if (gap_x == x && gap_y == y)//���Ǳ��������ǵײ���
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



