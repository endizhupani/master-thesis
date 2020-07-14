#include <iostream>
#include <string>
#include <exception>
using namespace std;

class E : public std::exception
{
    const char *msg;
    E(){};

public:
    explicit E(const char *s) throw() : msg(s) {}
    const char *what() const throw() { return msg; }
};

template <typename T>
class bwstack
{
private:
    static const int default_size = 10;
    static const int max_size = 1000;
    int _size;
    int _top;
    T *_stkptr;

public:
    explicit bwstack(int s = default_size);
    ~bwstack() { delete[] _stkptr; }
    T &push(const T &);
    T &pop();
    bool is_empty() const { return _top < 0; }
    bool is_full() const { return _top >= _size - 1; }
    int top() const { return _top; }
    int size() const { return _size; }
};

template <typename T>
bwstack<T>::bwstack(int s)
{
    if (s > max_size || s < 1)
        throw E("invalid stack size");
    else
        _size = s;

    _stkptr = new T[_size];
    _top = -1;
}

template <typename T>
T &bwstack<T>::push(const T &i)
{
    if (is_full())
        throw E("Stack is full");
    return _stkptr[++_top] = i;
}

template <typename T>
T &bwstack<T>::pop()
{
    if (is_empty())
        throw E("Stack is empty");
    return _stkptr[_top--];
}

int main(int argc, char **argv)
{
    try
    {
        bwstack<int> si(5);

        cout << "si size: " << si.size() << endl;
        cout << "si top: " << si.top() << endl;

        for (int i : {1, 2, 3, 4, 5})
        {
            si.push(i);
        }

        cout << "si top after pushes: " << si.top() << endl;
        cout << "si is " << (si.is_full() ? "" : "not ") << "full" << endl;

        while (!si.is_empty())
        {
            cout << "popped " << si.pop() << endl;
        }
    }
    catch (E &e)
    {
        cout << "Stack error: " << e.what() << endl;
    }

    try
    {
        bwstack<string> ss(5);

        cout << "si size: " << ss.size() << endl;
        cout << "si top: " << ss.top() << endl;

        for (string i : {"one", "two", "three", "four", "five"})
        {
            ss.push(i);
        }

        cout << "si top after pushes: " << ss.top() << endl;
        cout << "si is " << (ss.is_full() ? "" : "not ") << "full" << endl;

        while (!ss.is_empty())
        {
            cout << "popped " << ss.pop() << endl;
        }
    }
    catch (E &e)
    {
        cout << "Stack error: " << e.what() << endl;
    }
}
