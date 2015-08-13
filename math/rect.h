// Idioms
#pragma once

#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <math.h>

// Point

template <typename T>
class Point {
  public:
    Point() {}
    Point(T x, T y) { this->x = x; this->y = y; }
  public:
    typedef T type;
    T x;
    T y;
    bool operator==(const Point<T>& point);
    Point<T> operator+(const Point<T>& point);
};

template <typename T>
bool Point<T>::operator==(const Point<T>& point) {
  return x == point.x && y == point.y;
}

template <typename T>
Point<T> Point<T>::operator+(const Point<T>& point){
  return Point<T>(x + point.x, y + point.y);
}

template <typename T, typename ScaleType=T>
Point<T> operator+(const Point<T>& left, ScaleType a){
  return Point<T>(left.x + a, left.y + a);
}

template <typename T>
class Range : Point<T> {
public:
  Range() {}
  Range(T a, T b) : Point<T>(a, b) {}
  T width() const { return getB() - getA(); }
  T getA() const { return Point<T>::x; }
  T getB() const { return Point<T>::y; }
};

// Rect

template <typename T>
class Rect {
  public:
    Rect(){ common_construct(); }
    Rect(Point<T> _bl, Point<T> _tr) { bl = _bl; tr = _tr; }
    Rect(T b, T l, T t, T r) { bl = Point<T>(l, b);  tr = Point<T>(r, t); }
  protected:
    void common_construct() { bl = Point<T>(); tr = Point<T>(); }
  public:
    typedef T type;
    T GetBottom() { return bl.y; }
    T GetTop() { return tr.y; }
    T GetLeft() { return bl.x; }
    T GetRight() { return tr.x; }
    T GetWidth() const { return abs(tr.x - bl.x); }
    T GetHeight() const { return abs(tr.y - bl.y); }
    Point<T> bl; // point 1
    Point<T> tr; // point 2
    Rect get_intersect(const Rect& rect) {
      using namespace std;
      Rect intersect;
      intersect.bl = Point<T>(max(bl.x, rect.bl.x), max(bl.y, rect.bl.y));
      intersect.tr = Point<T>(min(tr.x, rect.tr.x), min(tr.y, rect.tr.y));
      return intersect;
    }
    bool intersects(const Rect& rect) {
      using namespace std;
      Rect intersect = get_intersect(rect);
      Point<T> bl = intersect.bl;
      Point<T> tr = intersect.tr;
      if (bl.x > tr.x || bl.y > tr.y )
        return false;
      return true;
    }
    Rect<T> operator+(const Rect<T>& rect);
};

template <typename T>
Rect<T> Rect<T>::operator+(const Rect<T>& rect) {
  return Rect<T>(bl + rect.bl, tr + rect.tr);
}


// Override ostream's << for Rect and Point parameters

template <typename T>
std::ostream& operator<<(std::ostream& out, const Point<T>& pt) {
  out << pt.x << " " << pt.y;
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Rect<T>& rect) {
  out << rect.bl << " " << rect.tr;
  return out;
}
