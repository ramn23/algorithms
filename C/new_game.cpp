#include "bits/stdc++.h" // For C++
#ifndef __GNUC__ // For GCC
#include <intrin.h>
#endif
#include <omp.h> // OpenMP library 
using namespace std; 
#define rep(i,n) for(int (i)=0;(i)<(int)(n);++(i))
#define rer(i,l,u) for(int (i)=(int)(l);(i)<=(int)(u);++(i))
#define reu(i,l,u) for(int (i)=(int)(l);(i)<(int)(u);++(i))
static const int INF = 0x3f3f3f3f; static const long long INFL = 0x3f3f3f3f3f3f3f3fLL;
typedef vector<int> vi; typedef pair<int, int> pii; typedef vector<pair<int, int> > vpii; typedef long long ll;
template<typename T, typename U> static void amin(T &x, U y) { if(y < x) x = y; }
template<typename T, typename U> static void amax(T &x, U y) { if(x < y) x = y; }
 \
#endif
}
inline int countOneBits(uint32_t x) {
#ifdef __GNUC__
	return __builtin_popcount(x);
#else
	return __popcnt(x);
#endif
}
inline int select32(uint32_t x, int k, int &popcnt) {
	uint32_t a, b, c; int t, s;
	a = (x & 0x55555555) + ((x >> 1) & 0x55555555);
	b = (a & 0x33333333) + ((a >> 2) & 0x33333333);
	c = (b & 0x0f0f0f0f) + ((b >> 4) & 0x0f0f0f0f);
	t = (c & 0xff) + ((c >> 8) & 0xff);
	popcnt = t + ((c >> 16) & 0xff) + ((c >> 24) & 0xff);
	if(popcnt <= k) return 32;
	s = 0;
	s += ((t - k - 1) & 128) >> 3; k -= t & ((t - k - 1) >> 8); //if(k >= t) s += 16, k -= t;
	t = (c >> s) & 0xf;
>> s) & 0x1;	s += ((t - k - 1) & 128) >> 4; k -= t & ((t - k - 1) >> 8); //if(k >= t) s += 8, k -= t;
	t = (b >> s) & 0x7;
x	t = (a >> s) & 0x3;
	s += ((t - k - 1) & 128) >> 6; k -= t & ((t - k - 1) >> 8); //if(k >= t) s += 2, k -= t;
	t = (x 
	s += ((t - k - 1) & 128) >> 7; //if(k >= t) s += 1;
	return s;
}
 
struct Mask128 {
	enum { HalfSize = 64 };
 
	uint64_t data[2];
 
	Mask128() {}
	explicit Mask128(uint64_t data0) : data{ data0, 0 } {}
	Mask128(uint64_t data0, uint64_t data1) : data{ data0, data1 } {}
 
	explicit operator bool() const {
		return (data[0] | data[1]) != 0;
	}
 
	bool get(unsigned pos) const {
		return data[pos / HalfSize] >> (pos % HalfSize) & 1;
	}
 
	void set(unsigned pos) {
		data[pos / HalfSize] |= uint64_t(1) << (pos % HalfSize);
	}
 
	void unset(unsigned pos) {
		data[pos / HalfSize] &= ~(uint64_t(1) << (pos % HalfSize));
	}
 
	Mask128 &operator&=(Mask128 that) {
		data[0] &= that.data[0];
		data[1] &= that.data[1];
		return *this;
	}
	Mask128 &operator|=(Mask128 that) {
		data[0] |= that.data[0];
		data[1] |= that.data[1];
		return *this;
	}
	Mask128 &operator^=(Mask128 that) {
		data[0] ^= that.data[0];
		data[1] ^= that.data[1];
		return *this;
	}
	Mask128 operator&(Mask128 that) const {
		return Mask128(data[0] & that.data[0], data[1] & that.data[1]);
	}
	Mask128 operator|(Mask128 that) const {
		return Mask128(data[0] | that.data[0], data[1] | that.data[1]);
	}
	Mask128 operator^(Mask128 that) const {
		return Mask128(data[0] ^ that.data[0], data[1] ^ that.data[1]);
	}
	Mask128 operator~() const {
		return Mask128(~data[0], ~data[1]);
	}
 
	Mask128 shiftRight1to63(uint8_t i) const {
		return Mask128(data[0] >> i | data[1] << (HalfSize - i), data[1] >> i);
	}
 
	Mask128 operator<<(uint8_t i) const {
		if(i == 0)
			return *this;
		else if(i < HalfSize)
			return Mask128(data[0] << i, data[1] << i | data[0] >> (HalfSize - i));
		else
			return Mask128(0, data[0] << (i - HalfSize));
	}
 
	Mask128 getLowestBit() const {
		if(data[0] != 0)
			return Mask128(data[0] & (~data[0] + 1), 0);
		else
			return Mask128(0, data[1] & (~data[1] + 1));
	}
 
	int getBitPos() const {
		if(data[0] != 0)
			return bsf(data[0]);
		else
			return HalfSize + bsf(data[1]);
	}
 
	int count() const {
		return popcount(data[0]) + popcount(data[1]);
	}
 
	int select(int k) const {
		int cntlo, dummy;
		int selectlo = select64(data[0], k, cntlo);
		if(selectlo < 64)
			return selectlo;
		else
			return 64 + select64(data[1], k - cntlo, dummy);
	}
 
	std::string toBitString() const {
		std::string res(128, '0');
		for(int i = 0; i < 128; ++ i)
			res[i] = '0' + get(i);
		return res;
	}
 
	static int bsf(uint64_t x) {
#if defined(__GNUC__)
		return __builtin_ctzll(x);
#elif defined(_M_X64) || defined(__amd64__)
		unsigned long res;
		_BitScanForward64(&res, x);
		return res;
#else
		uint32_t lo = (uint32_t)x;
		if(lo != 0)
			return findFirstBitPos(lo);
		else
			return 32 + findFirstBitPos((uint32_t)(x >> 32));
#endif
	}
 
	static int popcount(uint64_t x) {
#if defined(__GNUC__)
		return __builtin_popcountll(x);
#elif defined(_M_X64) || defined(__amd64__)
		return (int)__popcnt64(x);
#else
		uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
		return countOneBits(lo) + countOneBits(hi);
#endif
	}
 
	static int select64(uint64_t x, int k, int &respopcnt) {
		int cntlo, cnthi;
		int selectlo = select32((uint32_t)x, k, cntlo);
		int selecthi = select32((uint32_t)(x >> 32), k - cntlo, cnthi);
		respopcnt = cntlo + cnthi;
		return selectlo < 32 ? selectlo : 32 + selecthi;
	}
 
	bool operator==(const Mask128 &that) const {
		return data[0] == that.data[0] && data[1] == that.data[1];
	}
};
	return __builtin_popcount(x);
#else
	return __popcnt(x);
#endif
}
inline int select32(uint32_t x, int k, int &popcnt) {
	uint32_t a, b, c; int t, s;
	a = (x & 0x55555555) + ((x >> 1) & 0x55555555);
	b = (a & 0x33333333) + ((a >> 2) & 0x33333333);
	c = (b & 0x0f0f0f0f) + ((b >> 4) & 0x0f0f0f0f);
	t = (c & 0xff) + ((c >> 8) & 0xff);
	popcnt = t + ((c >> 16) & 0xff) + ((c >> 24) & 0xff);
	if(popcnt <= k) return 32;
	s = 0;
	s += ((t - k - 1) & 128) >> 3; k -= t & ((t - k - 1) >> 8); //if(k >= t) s += 16, k -= t;
	t = (c >> s) & 0xf;
>> s) & 0x1;	s += ((t - k - 1) & 128) >> 4; k -= t & ((t - k - 1) >> 8); //if(k >= t) s += 8, k -= t;
	t = (b >> s) & 0x7;
x
 
namespace std {
template<> struct hash<Mask128> {
	size_t operator()(const Mask128 &mask) const {
		size_t r = 0;
		r ^= hash1((uint32_t)mask.data[0], 797093307U);
		r ^= hash1((uint32_t)(mask.data[0] >> 32), 4227048661U);
		r ^= hash1((uint32_t)mask.data[1], 2453653089U);
		r ^= hash1((uint32_t)(mask.data[1] >> 32), 2811347015U);
		return r;
	}
 
	static uint32_t hash1(uint32_t x, uint32_t rnd) {
		return (uint32_t)((uint64_t)x * rnd >> 32);
	}
};
}
 
const int Height = 10, Width = 10;
const int NumFigures = 19;
const string figurePictures[NumFigures] = {
	"*", "* *", "**", "* * *", "***",
	"* * * *", "****", "* * * * *", "*****", "** **",
	"*** *** ***", "*** ..* ..*", "..* ..* ***", "*.. *.. ***", "*** *.. *..",
	"** *.", "** .*", ".* **", "*. **"
};
struct FigureData {
	int h, w;
	Mask128 masks[Height * Width];
	const char *picture;
	Mask128 regionMask;
	bool position0;
	vector<uint8_t> positions1;
	int neighborBase[Height * Width];
	Mask128 neighborMasks[Height * Width];
	uint8_t rowMasks[Height];
	uint8_t colCounts[Width];
};
FigureData figureData[NumFigures];
Mask128 rowLineMasks[Height], colLineMasks[Width];
 
inline pair<uint8_t, uint8_t> posToCoord(uint8_t pos) {
	return make_pair(pos / Width, pos % Width);
}
 
void initData() {
	rep(i, NumFigures) {
		stringstream ss(figurePictures[i]);
		string line;
		vector<string> lines;
		while(ss >> line) lines.push_back(line);
		FigureData &f = figureData[i];
		f.h = (int)lines.size();
		f.w = 0;
		f.position0 = false;
		f.positions1.clear();
		rep(i, Height)
			f.rowMasks[i] = 0;
		rep(j, Width)
			f.colCounts[j] = 0;
		vector<int> positions;
		Mask128 mask(0);
		rep(i, lines.size()) {
			amax(f.w, (int)lines[i].size());
			rep(j, lines[i].size()) if(lines[i][j] == '*') 
            {
				int pos = i * Width + j;
				positions.push_back(pos);
				if(i == 0 && j == 0)
					f.position0 = true;
				else
					f.positions1.push_back(pos);
				mask.set(pos);
				f.rowMasks[i] |= 1U << j;
				++ f.colCounts[j];
			}
		}
		f.regionMask = Mask128(0);
		rer(i, 0, Height - f.h) rer(j, 0, Width - f.w) {
			int offset = i * Width + j;
			f.regionMask.set(offset);
			f.masks[offset] = mask << offset;
 
			f.neighborBase[offset] = 0;
			f.neighborMasks[offset] = Mask128(0);
 
			for(int pd : positions) {
				int p = offset + pd;
				int y, x; tie(y, x) = posToCoord(p);
				static const int dy[4] = { 0, 1, 0, -1 }, dx[4] = { 1, 0, -1, 0 };
				for(int d = 0; d < 4; ++ d) {
					int yy = y + dy[d], xx = x + dx[d];
					if(yy < 0 || yy >= Height || xx < 0 || xx >= Width) {
						++ f.neighborBase[offset];
						continue;
					} else {
						int q = yy * Width + xx;
						f.neighborMasks[offset].set(q);
					}
				}
			}
		}
 
		f.picture = figurePictures[i].c_str();
	}
	Mask128 row(0);
	rep(j, Width)
		row.set(j);
	rep(i, Height)
		rowLineMasks[i] = row << (i * Width);
	Mask128 col(0);
	rep(i, Height)
		col.set(i * Width);
	rep(j, Width)
		colLineMasks[j] = col << j;
}
 
struct Board {
	Mask128 mask;
 
	Board() : mask(0) {}
	explicit Board(Mask128 mask) : mask(mask) {}
 
	bool canPlace(const FigureData &f, int pos) const {
		int y, x; tie(y, x) = posToCoord(pos);
		assert(0 <= y && 0 <= x && y + f.h <= Height && x + f.w <= Width);
		return !(f.masks[pos] & mask);
	}
 
	unsigned place(const FigureData &f, int pos) {
		assert(canPlace(f, pos));
		mask |= f.masks[pos];
 
		int y, x; tie(y, x) = posToCoord(pos);
		unsigned rows = 0, cols = 0;
		for(int i = y; i < y + f.h; ++ i)
			if(checkRow(i))
				rows |= 1U << i;
		for(int j = x; j < x + f.w; ++ j)
			if(checkCol(j))
				cols |= 1U << j;
		unsigned cleared = rows | cols << Height;
		while(rows) {
			int i = findFirstBitPos(rows);
			mask &= ~rowLineMasks[i];
			rows ^= 1U << i;
		}
		while(cols) {
			int j = findFirstBitPos(cols);
			mask &= ~colLineMasks[j];
			cols ^= 1U << j;
		}
		if(!mask)
			cleared |= 1U << (Height + Width);
		return cleared;
	}
 
	bool checkRow(int i) const {
		return !(~mask & rowLineMasks[i]);
	}
	bool checkCol(int j) const {
		return !(~mask & colLineMasks[j]);
	}
 
	static int calcScore(const FigureData &f, unsigned cleared) {
		int r = f.h * f.w;
		if(cleared) {
			int x = countOneBits(cleared & ((1U << Height) - 1));
			int y = countOneBits(cleared >> Height & ((1U << Width) - 1));
			r += x * x + y * y + 5 * x * y;
			if(cleared >> (Height + Width) & 1)
				r += 500;
		}
		return r;
	}
 
	Mask128 getPlaceableRegionFor(const FigureData &f) const {
		Mask128 space = ~mask;
		Mask128 res = f.regionMask;
		if(f.position0) res &= space;
		for(uint8_t pos : f.positions1)
			res &= space.shiftRight1to63(pos);
		return res;
	}
 
	void show(ostream &os, Mask128 prevMask = Mask128(0)) const {
		if(!os) return;
		rep(i, Height) {
			rep(j, Width)
				os << (mask.get(i * Width + j) ? (prevMask.get(i * Width + j) ? '*' : '@') : '.');
			os << '\n';
		}
	}
};
 
vector<double> figureProbability;
mt19937 globalRandomEngine, globalUnreproducibleRandomEngine;
discrete_distribution<int> figureDistribution;
 
struct LinearCoeffs {
	static const int constant;
	static const int vec[6];
	static const uint8_t canonMap[1 << 4];
 
	int coeffs[1 << 4];
 
	LinearCoeffs() {
		rep(s, 1 << 4)
			coeffs[s] = vec[canonMap[s]];
	}
};
const int LinearCoeffs::constant = 600614625;	//6.00614625e+01
const int LinearCoeffs::vec[6] = {
	//	3.58594440e-01, -3.18756518e-01, 6.52172873e-03, -2.64978541e+00, -6.51681865e-01, 0
	3585944, -3187565, 65217, -26497854, -6516819, 0
};
const uint8_t LinearCoeffs::canonMap[1 << 4] = {
	0, 1, 1, 2, 1, 2, 3, 4, 1, 3, 2, 4, 2, 4, 4, 5
};
 
const LinearCoeffs linearCoeffs;
double linearEval(const Board &board) {
	int res = linearCoeffs.constant;
	uint32_t curRow = (1U << (Width + 2)) - 1;
	const Mask128 mask = board.mask;
	rer(i, -1, Height - 1) {
		const uint8_t shift = (i + 1) * Width;
		uint32_t nextRow =
			i == Height - 1 ? (1U << (Width + 2)) - 1 :
			uint32_t((
				shift == 0 ? mask.data[0] :
				shift < 64 ?
				mask.data[0] >> shift | mask.data[1] << (64 - shift) :
				mask.data[1] >> (shift - 64)) & ((1U << Width) - 1))
			<< 1 | 1U << 0 | 1U << (Width + 1);
		uint32_t x = curRow, y = nextRow;
		rer(j, -1, Width - 1) {
			uint32_t bits = (x & 3) | (y & 3) << 2;
			res += linearCoeffs.coeffs[bits];
			x >>= 1, y >>= 1;
		}
		curRow = nextRow;
	}
	return res * 1e-7;
}
 
class FastLinearEval {
public:
	void init(const Board &board) {
		_mask = board.mask;
 
		rep(j, Width)
			_colCounts[j] = 0;
 
		rep(i, Height) {
			const uint8_t shift = i * Width;
			const uint16_t rowMask = uint16_t((
				shift == 0 ? _mask.data[0] :
				shift < 64 ?
				_mask.data[0] >> shift | _mask.data[1] << (64 - shift) :
				_mask.data[1] >> (shift - 64)) & ((1U << Width) - 1));
 
			_rows[i] = rowMask << 1 | EmptyRowMask;
			_borderedRows[i + 1] = _rows[i];
 
			for(uint32_t x = rowMask; x != 0; x &= x - 1) {
				int col = findFirstBitPos(x);
				++ _colCounts[col];
			}
		}
		_borderedRows[0] = _borderedRows[Height + 1] = FilledRowMask;
 
		_originalSum = linearCoeffs.constant;
		for(int y = -1; y <= Height - 1; ++ y) {
			const uint32_t curRow = _borderedRows[y + 1];
			const uint32_t nextRow = _borderedRows[y + 2];
			for(int x = -1; x <= Width - 1; ++ x) {
				_bits[y + 1][x + 1] = ((curRow >> (x + 1)) & 3) | ((nextRow >> (x + 1)) & 3) << 2;
				_originalSum += linearCoeffs.coeffs[_bits[y + 1][x + 1]];
			}
		}
	}
 
	double evalPlace(const FigureData &f, int pos, bool isLast = false) const {
		int posY, posX;
		tie(posY, posX) = posToCoord(pos);
		array<uint16_t, Height> newRows = _rows;
 
		unsigned clearRows, clearCols;
		tie(clearRows, clearCols) = placeAndClear(f, posY, posX, newRows.data(), _colCounts.data());
 
		int loY = posY, hiY = posY + f.h;
		if(clearCols != 0)
			loY = 0, hiY = Height;
 
		int sum = _originalSum;
		uint32_t curRow = loY == 0 ? FilledRowMask : newRows[loY - 1];
		uint32_t curRowDiff = 0;
		for(int y = loY - 1; y < hiY; ++ y) {
			const uint32_t nextRow = y == Height - 1 ? FilledRowMask : newRows[y + 1];
			const uint32_t nextRowDiff = _borderedRows[y + 2] ^ nextRow;
			const uint32_t diffCols = curRowDiff | nextRowDiff;
			const uint32_t diffs = diffCols | diffCols >> 1;
			for(uint32_t ds = diffs; ds != 0; ds &= ds - 1) {
				int x = findFirstBitPos(ds) - 1;
				uint32_t newBits = ((curRow >> (x + 1)) & 3) | ((nextRow >> (x + 1)) & 3) << 2;
				sum += linearCoeffs.coeffs[newBits];
				sum -= linearCoeffs.coeffs[_bits[y + 1][x + 1]];
			}
			curRow = nextRow;
			curRowDiff = nextRowDiff;
		}
 
		double score = sum * 1e-7;
 
		if(isLast) {
			bool no7 = true;
			for(int y = 0; y <= Height - 1; ++ y) {
				uint32_t a = ~newRows[y] & FilledRowMask;
				if((a & (a >> 1) & (a >> 2) & (a >> 3) & (a >> 4)) != 0) {
					no7 = false;
					break;
				}
			}
			if(no7) score -= 1000;
 
			bool no8 = true;
			for(int y = 0; y <= Height - 5; ++ y) {
				if((~newRows[y] & ~newRows[y + 1] & ~newRows[y + 2] & ~newRows[y + 3] & ~newRows[y + 4] & FilledRowMask) != 0) {
					no8 = false;
					break;
				}
			}
			if(no8) score -= 1000;
 
			int pos10y = -1, pos10x = -1;
			for(int y = 0; y <= Height - 3; ++ y) {
				uint32_t emptyCols = ~newRows[y] & ~newRows[y + 1] & ~newRows[y + 2] & FilledRowMask;
				uint32_t region = emptyCols & (emptyCols >> 1) & (emptyCols >> 2);
				if(region != 0) {
					pos10y = y;
					pos10x = findFirstBitPos(region) - 1;
					break;
				}
			}
 
			if(pos10y == -1) {
				score -= 1000;
			} else {
				array<uint8_t, Width> newColCounts = _colCounts;
				rep(j, f.w)
					newColCounts[posX + j] += f.colCounts[j];
				if(clearRows) {
					int p = countOneBits(clearRows);
					rep(j, Width) newColCounts[j] -= p;
				}
				for(unsigned t = clearCols; t != 0; t &= t - 1)
					-- newColCounts[findFirstBitPos(t)];
 
				placeAndClear(figureData[10], pos10y, pos10x, newRows.data(), newColCounts.data());
 
				bool notwo10s = true;
				for(int y = 0; y <= Height - 3; ++ y) {
					uint32_t emptyCols = ~newRows[y] & ~newRows[y + 1] & ~newRows[y + 2] & FilledRowMask;
					if((emptyCols & (emptyCols >> 1) & (emptyCols >> 2)) != 0) {
						notwo10s = false;
						break;
					}
				}
				if(notwo10s)
					score -= 500;
			}
		}
 
		return score;
	}
 
private:
	static pair<unsigned, unsigned> placeAndClear(const FigureData &f, int posY, int posX, uint16_t newRows[Height], const uint8_t colCounts[Width]) {
		unsigned clearRows = 0, clearCols = 0;
		rep(i, f.h) {
			newRows[posY + i] |= f.rowMasks[i] << (posX + 1);
			if(newRows[posY + i] == FilledRowMask) {
				newRows[posY + i] = EmptyRowMask;
				clearRows |= 1U << (posY + i);
			}
		}
		rep(j, f.w) {
			if(colCounts[posX + j] + f.colCounts[j] == Width) {
				uint32_t mask = ~(1U << (posX + j + 1));
				rep(i, Height)
					newRows[i] &= mask;
				clearCols |= 1U << (posX + j);
			}
		}
		return make_pair(clearRows, clearCols);
	}
 
	enum : uint16_t {
		EmptyRowMask = 1U << 0 | 1U << (Width + 1),
		FilledRowMask = (1U << (Width + 2)) - 1
	};
	Mask128 _mask;
	array<uint16_t, Height> _rows;
	uint32_t _borderedRows[Height + 2];
	array<uint8_t, Width> _colCounts;
	uint8_t _bits[Height + 1][Width + 1];
	int _originalSum;
};
 
struct Xor128 {
	unsigned x, y, z, w;
	Xor128() : x(123456789), y(362436069), z(521288629), w(88675123) {}
	unsigned operator()() {
		unsigned t = x ^ (x << 11);
		x = y; y = z; z = w;
		return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
	}
	unsigned operator()(unsigned n) { return operator()() % n; }
};
 
#ifdef _WIN32
extern "C" int __stdcall QueryPerformanceFrequency(long long*);
extern "C" int __stdcall QueryPerformanceCounter(long long*);
double getTime() {
	long long c, freq;
	QueryPerformanceCounter(&c);
	QueryPerformanceFrequency(&freq);
	return c * 1. / freq;
}
#else
#include <sys/time.h>
double getTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1e6;
}
#endif
 
#ifdef MY_LOCAL_RUN
 
#if defined(_WIN32) && !defined(_WINDOWS_)
struct FILETIME {
	unsigned dwLowDateTime, dwHighDateTime;
};
extern "C" void* __stdcall GetCurrentProcess(void);
extern "C" int __stdcall GetProcessTimes(void *hProcess, FILETIME *lpCreationTime, FILETIME *lpExitTime, FILETIME *lpKernelTime, FILETIME *lpUserTime);
#endif
#ifndef _WIN32
#include <sys/time.h>
#include <sys/resource.h>
#endif
 
static void getCPUTime(double &userTime, double &systemTime) {
#ifdef _WIN32
	void *handle = GetCurrentProcess();
	FILETIME dummy1, dummy2, kernel, user;
	GetProcessTimes(handle, &dummy1, &dummy2, &kernel, &user);
	userTime = user.dwHighDateTime * 429.4967296 + user.dwLowDateTime * 1e-7;
	systemTime = kernel.dwHighDateTime * 429.4967296 + kernel.dwLowDateTime * 1e-7;
#else
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	userTime = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
	systemTime = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
#endif
}
static double getCPUTime() {
	double user, sys;
	getCPUTime(user, sys);
	return user + sys;
}
 
struct CPUTimeIt {
	double user, sys, real;
	const char *msg;
 
	CPUTimeIt(const char *msg_) : msg(msg_) {
		getCPUTime(user, sys);
		real = getTime();
	}
	~CPUTimeIt() {
		double userEnd, sysEnd;
		getCPUTime(userEnd, sysEnd);
		double realEnd = getTime();
		fprintf(stderr, "%s: user %.6fs / sys %.6fs / real %.6fs\n",
			msg, userEnd - user, sysEnd - sys, realEnd - real);
	}
 
	operator bool() { return false; }
};
#define CPUTIMEIT(s) if(CPUTimeIt cputimeit_##__LINE__ = s); else
 
#endif
 
struct Move1 {
	double score;
	int figure;
	int pos;
	Move1() : score(-1e99), figure(-1), pos(-1) {}
	Move1(double score, int figure, int pos) : score(score), figure(figure), pos(pos) {}
	bool isNoMove() const { return score == -1e99; }
	bool update(double s, int f, int p) {
		if(score < s) {
			score = s;
			figure = f;
			pos = p;
			return true;
		} else {
			return false;
		}
	}
};

struct Move3 {
	double score;
	Move1 moves[3];
	Move3() : score(-1e99) {}
	bool isNoMove() const { return score == -1e99; }
	bool update(double s, const Move1 &first, const Move1 &second, const Move1 &third) {
		if(score < s) {
			score = s;
			moves[0] = first;
			moves[1] = second;
			moves[2] = third;
			return true;
		} else {
			return false;
		}
	}
};
 
double calcTotalScore(const Move1 &firstMove, const Move1 &secondMove, const Move1 &thirdMove) {
	double totalScore = 0;
	totalScore += firstMove.score / 4;
	totalScore += secondMove.score / 2;
	totalScore += thirdMove.score * 25;
	return totalScore;
}
 
template<typename Proc>
void enumMovesTemplate(const Board &board, int fi, bool isLast, Proc proc) {
	const FigureData &f = figureData[fi];
	FastLinearEval fastLinearEval;
	fastLinearEval.init(board);
	Mask128 placeable = board.getPlaceableRegionFor(f);
	while(placeable) {
		int pos = placeable.getBitPos();
 
		double score = fastLinearEval.evalPlace(f, pos, isLast);
 
		proc(score, fi, pos);
 
		placeable.unset(pos);
	}
}
 
void enumMoves(const Board &board, int fi, vector<Move1> &res, bool isLast) {
	enumMovesTemplate(board, fi, isLast, [&res](double score, int fi, int pos) {
		res.emplace_back(score, fi, pos);
	});
}
 
Move1 findBestMove1(const Board &board, int fi, bool isLast) {
	Move1 bestMove;
	enumMovesTemplate(board, fi, isLast, [&bestMove](double score, int fi, int pos) {
		bestMove.update(score, fi, pos);
	});
	return bestMove;
}
 
void findBestMoveFastest(Move3 &bestMove, const Board &initBoard, const int sortedNextFigures[3]) {
	Move1 firstMove;
	int bestFirstIndex = -1;
	rep(firstIndex, 3) if(firstIndex == 2) {
		Move1 move = findBestMove1(initBoard, sortedNextFigures[firstIndex], false);
		if(bestMove.score < move.score) {
			firstMove = move;
			bestFirstIndex = firstIndex;
		}
	}
 
	if(firstMove.isNoMove())
		return;
 
	Board secondBoard = initBoard;
	secondBoard.place(figureData[firstMove.figure], firstMove.pos);
 
	Move1 secondMove;
	int bestSecondIndex = -1;
	rep(secondIndex, 3) if(secondIndex != bestFirstIndex) {
		Move1 move = findBestMove1(secondBoard, sortedNextFigures[secondIndex], false);
		if(secondMove.score < move.score) {
			secondMove = move;
			bestSecondIndex = secondIndex;
		}
	}
 
	if(secondMove.isNoMove())
		return;
 
	Board thirdBoard = secondBoard;
	thirdBoard.place(figureData[secondMove.figure], secondMove.pos);
 
	int thirdIndex = 0;
	while(thirdIndex == bestFirstIndex || thirdIndex == bestSecondIndex) ++ thirdIndex;
 
	Move1 thirdMove = findBestMove1(thirdBoard, sortedNextFigures[thirdIndex], true);
 
	if(thirdMove.isNoMove())
		return;
 
	bestMove.update(calcTotalScore(firstMove, secondMove, thirdMove), firstMove, secondMove, thirdMove);
}
 
void cutoffMoveCandidates(vector<Move1> &candidates, double cutoffRatio) {
	double maxScore = -1e99;
	for(const Move1 &move : candidates)
		amax(maxScore, move.score);
	candidates.erase(remove_if(candidates.begin(), candidates.end(), [maxScore, cutoffRatio](const Move1 &a) {
		return a.score < maxScore * cutoffRatio;
	}), candidates.end());
}
 
 
Move3 findBestMoveGeneralized(Move3 &bestMove, Board initBoard, const int sortedNextFigures[3], const array<double, 3> cutOffRatios) {
	struct FirstTwoMoves {
		double score;
		Move1 firstMove, secondMove;
		int thirdIndex;
	};
 
	vector<Move1> firstMoveCandidatesList[3];
	vector<Move1> secondMoveCandidates;
 
	rep(firstIndex, 3) {
		int firstfi = sortedNextFigures[firstIndex];
		enumMoves(initBoard, firstfi, firstMoveCandidatesList[firstIndex], false);
	}
 
	const double firstCutoffRatio = cutOffRatios[0];
	const double secondCutoffRatio = cutOffRatios[1];
	const double firstTwoCutoffRatio = cutOffRatios[2];
 
	vector<int> indexPerm = { 0, 1, 2 };
	static unordered_map<Mask128, FirstTwoMoves> firstTwoMoves;
	firstTwoMoves.clear();
	do {
		vector<Move1> firstMoveCandidates = firstMoveCandidatesList[indexPerm[0]];
 
		cutoffMoveCandidates(firstMoveCandidates, firstCutoffRatio);
		for(const Move1 &firstMove : firstMoveCandidates) {
			Board secondBoard = initBoard;
			secondBoard.place(figureData[firstMove.figure], firstMove.pos);
 
			secondMoveCandidates.clear();
			enumMoves(secondBoard, sortedNextFigures[indexPerm[1]], secondMoveCandidates, false);
 
			cutoffMoveCandidates(secondMoveCandidates, secondCutoffRatio);
			for(const Move1 &secondMove : secondMoveCandidates) {
				Board thirdBoard = secondBoard;
				thirdBoard.place(figureData[secondMove.figure], secondMove.pos);
 
				auto &firstTwoVal = firstTwoMoves.emplace(make_pair(thirdBoard.mask, FirstTwoMoves{ -1e99 })).first->second;
				double firstTwoScore = firstMove.score / 4 + secondMove.score / 2;
				if(firstTwoVal.score < firstTwoScore) {
					firstTwoVal.score = firstTwoScore;
					firstTwoVal.firstMove = firstMove;
					firstTwoVal.secondMove = secondMove;
					firstTwoVal.thirdIndex = indexPerm[2];
				}
			}
		}
	} while(next_permutation(indexPerm.begin(), indexPerm.end()));
 
	double maxFirstTwoScore = -1e99;
	for(const auto &firstTwo : firstTwoMoves)
		amax(maxFirstTwoScore, firstTwo.second.score);
 
	for(const auto &firstTwo : firstTwoMoves) {
		const auto &firstTwoVal = firstTwo.second;
		if(firstTwoVal.score < maxFirstTwoScore * firstTwoCutoffRatio)
			continue;
 
		Board thirdBoard(firstTwo.first);
 
		Move1 thirdMove = findBestMove1(thirdBoard, sortedNextFigures[firstTwoVal.thirdIndex], true);
		if(thirdMove.isNoMove()) continue;
 
		double totalScore = calcTotalScore(firstTwoVal.firstMove, firstTwoVal.secondMove, thirdMove);
 
		bestMove.update(totalScore, firstTwoVal.firstMove, firstTwoVal.secondMove, thirdMove);
	}
	return bestMove;
}
 
Move3 findBestMove(Board initBoard, const int sortedNextFigures[3], double remainingTime) {
	Move3 bestMove;
 
	findBestMoveFastest(bestMove, initBoard, sortedNextFigures);
	if(!bestMove.isNoMove() && bestMove.score >= 1700)
		return bestMove;
 
	findBestMoveGeneralized(bestMove, initBoard, sortedNextFigures,
	{ 0.99, 1.0, 0.0 });
 
	if(!bestMove.isNoMove() && bestMove.score >= 1500 - max(0., 2.0 - remainingTime) * 100)
		return bestMove;
 
	findBestMoveGeneralized(bestMove, initBoard, sortedNextFigures,
	{ 0.8, 0.99, 0.0 });
 
	if(!bestMove.isNoMove() && bestMove.score >= 1350 - max(0., 2.0 - remainingTime) * 75)
		return bestMove;
 
	findBestMoveGeneralized(bestMove, initBoard, sortedNextFigures,
	{ 0.0, 0.0, 0.0 });
 
	return bestMove;
}
 
int mymain() {
#ifdef MY_LOCAL_RUN
#define RANDOM_TEST
#endif
#ifdef RANDOM_TEST
	ofstream logf;// ("newgame_log.txt", ios_base::app);
	long long totalTotalScore = 0;
	const int GAMES = 100;
#else
	const int GAMES = 1;
#endif
	for(int games = 0; games < GAMES; ++ games) {
#ifdef RANDOM_TEST
		mt19937 testGenerationEngine(100000 + games);
		globalRandomEngine.seed(100000 + games + 1);
		logf << "new game " << games << endl;
#else
		globalUnreproducibleRandomEngine.seed(random_device{}());
#endif
		double startTime = getTime(), timeLimit = startTime + 5.135;
 
		int turns = 0;
		long long totalScore = 0;
		Board curBoard;
		Mask128 prevBoardMask(0);
		vector<double> scoreLog;
		vector<Board> boardLog;
		array<int, 3> originalNextFigures;
#ifdef RANDOM_TEST
		curBoard.show(logf, prevBoardMask);
#	ifdef MY_LOCAL_RUN
		CPUTIMEIT("turns") for(; turns < 50000; ++ turns) {
#	else
		for(; turns < 50000; ++ turns) {
#	endif
			boardLog.push_back(curBoard);
			rep(k, 3)
				originalNextFigures[k] = figureDistribution(testGenerationEngine);
			logf << "next: ";
			rep(k, 3) {
				if(k != 0) logf << " | ";
				logf << figureData[originalNextFigures[k]].picture;
			}
			logf << '\n';
#else
		while(scanf("%d%d%d", &originalNextFigures[0], &originalNextFigures[1], &originalNextFigures[2]) == 3 && originalNextFigures[0] != -1) {
			rep(k, 3) -- originalNextFigures[k];
#endif
			double remainingTime = timeLimit - getTime();
 
			array<int, 3> sortedNextFigures = originalNextFigures;
			sort(sortedNextFigures.begin(), sortedNextFigures.end());
 
			Move3 bestMove;
			if(remainingTime > 0) {
				bestMove = findBestMove(curBoard, sortedNextFigures.data(), remainingTime);
			} else {
#ifdef RANDOM_TEST
				cerr << "no remaining time!" << endl;
				logf << "no remaining time!" << endl;
#endif
			}
 
			if(bestMove.isNoMove()) {
#ifndef RANDOM_TEST
				puts("-1 -1 -1 -1 -1 -1 -1 -1 -1");
				fflush(stdout);
				continue;
#else
				if(remainingTime > 0) {
					{
						cerr << "last scores:";
						int num = min((int)scoreLog.size(), 5);
						rep(k, num)
							cerr << " " << scoreLog[scoreLog.size() - num + k];
						cerr << endl;
					}
					{
						cerr << "board eval:";
						int num = min((int)boardLog.size(), 5);
						rep(k, num)
							cerr << " " << linearEval(boardLog[boardLog.size() - num + k]);
						cerr << endl;
					}
					curBoard.show(cerr);
					cerr << "next: ";
					rep(k, 3) {
						if(k != 0) cerr << " | ";
						cerr << figureData[originalNextFigures[k]].picture;
					}
					cerr << '\n';
				}
				break;
#endif
			}
 
#ifdef RANDOM_TEST
			scoreLog.push_back(bestMove.score);
			logf << "best: " << bestMove.score << '\n';
#endif
			rep(k, 3) {
				Move1 move = bestMove.moves[k];
				int i = (int)(find(originalNextFigures.begin(), originalNextFigures.end(), move.figure) - originalNextFigures.begin());
				assert(i < 3);
				originalNextFigures[i] = -1;
#ifndef RANDOM_TEST
				if(k != 0) putchar(' ');
				printf("%d %d %d", i + 1,
					move.pos / 10 + figureData[move.figure].h, move.pos % 10 + 1);
#endif
				unsigned cleared = curBoard.place(figureData[move.figure], move.pos);
				totalScore += Board::calcScore(figureData[move.figure], cleared);
 
#ifdef RANDOM_TEST
				if(k != 0) logf << '\n';
				curBoard.show(logf, prevBoardMask);
				prevBoardMask = curBoard.mask;
#endif
			}
#ifndef RANDOM_TEST
			puts("");
			fflush(stdout);
#endif
		}
#ifdef RANDOM_TEST
		logf << "score = " << totalScore << ", turns = " << turns * 3 << endl << endl;
		cerr << "score = " << totalScore << ", turns = " << turns * 3 << endl;
		totalTotalScore += totalScore;
#endif
		}
 
#ifdef RANDOM_TEST
	cerr << "total total score = " << totalTotalScore << endl;
	logf << "total total score = " << totalTotalScore << endl;
#endif
	return 0;
		}
 
int main() {
	initData();
	{
		figureProbability = vector<double>(NumFigures, 1.0);
		//rep(i, 5)
		//	figureProbability[i] = 0;
		//rep(i, 4)
		//	figureProbability[15 + i] = 0;
		figureDistribution = discrete_distribution<int>(figureProbability.begin(), figureProbability.end());
	}
#ifdef MY_LOCAL_RUN
	//return bench::benchPlayout(), 0;
	//return minmaxSearch(), 0;
	//return bench::benchPlaceAndLinearEval(), 0;
	//return bench::benchCheckMaximalFigures(), 0;
#endif
	return mymain();
}
