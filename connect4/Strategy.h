/********************************************************
*	Strategy.h : ���Խӿ��ļ�                           *
*	������                                              *
*	zhangyf07@gmail.com                                 *
*	2010.8                                              *
*********************************************************/

#ifndef STRATEGY_H_
#define	STRATEGY_H_

#include "Point.h"
#include <iostream>
#include<vector>
#include<cassert>
#include <ctime>
#include <iostream>
#include "Judge.h"
#include <ctime>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <algorithm>
using namespace std;

const int INF = 1000;

extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board,
	const int lastX, const int lastY, const int noX, const int noY);

extern "C" __declspec(dllexport) void clearPoint(Point* p);

void clearArray(int M, int N, int** board);

int search(int** board, int * cursor, const int M, const int N, const int noX, const int noY, int depth, int player, int opponent, int column, int alpha, int beta);
/*
	board: ���̵Ķ�ά�����ʾ
	cursor : ����λ�õ��α�
	M,N:���̴�С
	noX, noY : �����ϵĲ������ӵ�
	depth:�������
	player:��ǰ��ɫ���
	opponent�����ֱ��
	column:�б��
	alpha:alphaֵ
	beta:betaֵ
*/

int evaluate(const int M, const int N, int** board, int noX, int noY, int * cursor, int player);
/*
	board: ���̵Ķ�ά�����ʾ
	M,N:���̴�С
	noX, noY : �����ϵĲ������ӵ�
	depth:�������
	player:��ǰ��ɫ���
*/

int horizontal(int ** board, const int M, const int N, int * t, int player, int noX, int noY, int x, int y, int gap_y, int interspace, int count);
int main_diagonal(int ** board, const int M, const int N, int * t, int player, int noX, int noY, int x, int y, int gap_x, int gap_y, int interspace, int count);
int secondary_diagonal(int ** board, const int M, const int N, int * t, int player, int noX, int noY, int x, int y, int gap_x, int gap_y, int interspace, int count);

#endif