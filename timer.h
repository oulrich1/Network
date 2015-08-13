#pragma once

// C
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// C++
#include <iostream>
#include <vector>

///  ---  B E G I N   T I M E R  ---  ///

template <typename T>
T ClockToSeconds(clock_t c) {
	return (T(c) / CLOCKS_PER_SEC);
}

template <typename TimeType>
class Timer {
private:
	clock_t _t1;
	clock_t _t2;
	TimeType m_time_sec;
	TimeType mTotalTime;
	bool m_bDidCalcTime;

public:
	Timer() {
		_t1 = clock();
		_t2 = clock();
		m_bDidCalcTime = false;
		m_time_sec = TimeType(0.0);
		mTotalTime = TimeType(0.0);
	}
	typedef enum {
		TIME_UNKNOWN,
		TIME_START,
		TIME_STOP
	} When;

	void start() {
		_t1 = clock();
		setDidCalcTime(false);
	}

	TimeType stop() {
		_t2 = clock();
		setDidCalcTime(false);
		return getTime();
	}

	TimeType getTime() {
		if (!didCalcTime()) {
			calcTime();
			updateTotalTime();
		}
		return m_time_sec;
	}

	inline TimeType getTotalTime() const {
		return mTotalTime;
	}

	inline bool didCalcTime() const { return m_bDidCalcTime; }

	inline clock_t getClockDiff() {
		return getClock(TIME_STOP) - getClock(TIME_START);
	}

	clock_t getClock(When w = TIME_UNKNOWN) {
		switch (w) {
			case TIME_START: return _t1;
			case TIME_STOP:  return _t2;
			default: return clock();
		}
	}
private:
	inline void updateTotalTime() {
		mTotalTime += m_time_sec;
	}
	void calcTime() {
		setTime(ClockToSeconds<TimeType>(getClockDiff()));
		setDidCalcTime(true);
	}

	void setDidCalcTime(bool bDidCalc) {
		m_bDidCalcTime = bDidCalc;
	}
	void setTime(TimeType t) {
		m_time_sec = t;
	}
};

#define TIME(func)            \
([&]() -> float {              \
  Timer<float> timer;         \
  timer.start();              \
  func;                       \
  timer.stop();               \
  return timer.getTime();     \
})();

#define TIME_ASSIGN(assign, func)    \
([&]() -> float {        \
  Timer<float> timer;         \
  timer.start();              \
  assign = func;              \
  timer.stop();               \
  return timer.getTime();     \
})();
