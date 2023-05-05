void zero_division_016 () {
    int dividend = 1000 ;
    int divisor1;
    const int v1 = 5, v2 = 6;
    divisor1 = dividend / 5;
	int a = test1(3, 5);
	int b1=2, b2=3;
	int b = test1(b1, b2);
    	printf("%d", divisor1);
	int data=0;
	int buffer[10] = { 0 };
	if (data >= 0)
	{
		buffer[data] = 1;
		/* Print the array values */
		for(i = 0; i < 10; i++)
		{
			printIntLine(buffer[i]);
		}
	}
	else
	{
		printLine("ERROR: Array index is negative.");
    }
}

int test1(int a, int b) {
	return a+b;
}

int main() {
    zero_division_016();
}

