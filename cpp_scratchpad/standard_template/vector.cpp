#include <iostream>
#include <cstring>
#include <vector>

using namespace std;

int main()
{
    cout << "Vector from initialier list: " << endl;

    vector<int> vil = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    cout << "size: " << vil.size() << endl;
    cout << "front: " << vil.front() << endl;
    cout << "back: " << vil.back() << endl;

    cout << endl
         << "Iterator:" << endl;

    auto it_begin = vil.begin();
    auto it_end = vil.end();

    for (auto i = it_begin; i < it_end; ++i)
    {
        cout << *i << " ";
    }

    cout << endl;

    cout << endl
         << "INdex:" << endl;
    cout << "element at 5:" << vil[5] << endl;
    cout << "element at 5: " << vil.at(5) << endl;

    

    return 0;
}