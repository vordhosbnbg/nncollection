#pragma once

template<class T>
class NormalizedValue
{
public:

    NormalizedValue():
        minVal(0),
        maxVal(1),
        minValNormalized(0),
        maxValNormalized(1),
        rangeVal(maxVal - minVal),
        reciprocalRangeVal((maxValNormalized - minValNormalized) / (maxVal - minVal)),
        normalizedValue(0)
    {

    }

    NormalizedValue(T min, T max, T minNormalized, T maxNormalized) :
        minVal(min),
        maxVal(max),
        minValNormalized(minNormalized),
        maxValNormalized(maxNormalized),
        rangeVal((maxVal - minVal) / (maxValNormalized - minValNormalized)),
        reciprocalRangeVal((maxValNormalized - minValNormalized) / (maxVal - minVal)),
        normalizedValue(0)
    {

    }

    ~NormalizedValue() = default;

    void updateMinMax()
    {
        rangeVal= maxVal - minVal;
        reciprocalRangeVal = (maxValNormalized - minValNormalized) / (maxVal - minVal);
    }

    T get() const
    {
        T retVal = static_cast<T>((normalizedValue - minValNormalized)/reciprocalRangeVal + minVal);
        return retVal;
    }

    void set(T val)
    {
        if(val < minVal)
        {
            val = minVal;
        }
        else if(val > maxVal)
        {
            val = maxVal;
        }
        normalizedValue = (val - minVal)*reciprocalRangeVal + minValNormalized;
    }

    double getNormalized() const
    {
        return normalizedValue;
    }

    void setNormalized(double val)
    {
        normalizedValue = val;
    }

    T getMin() const
    {
        return minVal;
    }

    void setMin(T val)
    {
        minVal = val;
        updateMinMax();
    }

    T getMax() const
    {
        return maxVal;
    }

    void setMax(T val)
    {
        maxVal = val;
        updateMinMax();
    }


private:
    T minVal;
    T maxVal;
    T minValNormalized;
    T maxValNormalized;
    T rangeVal;
    T reciprocalRangeVal;
    double normalizedValue;
};
