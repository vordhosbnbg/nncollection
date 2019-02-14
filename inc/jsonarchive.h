#pragma once
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <fstream>
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h>

template<typename T>
class ValueRestorer
{
public:
    ValueRestorer(T& obj) : _obj(obj), _value(obj){}
    ~ValueRestorer()
    {
        _obj = _value;
    }

    T getInitialValue() const
    {
        return _value;
    }


private:
    T& _obj;
    T _value;
};

class JSONArchive
{
public:
    JSONArchive(const std::string& filename): _filename(filename) {}



    template<typename T>
    void save(const std::string& key, T& obj)
    {
        if(_writerPtr)
        {
            _writerPtr->Key(key.c_str());
            save(obj);
        }
    }

//    template<typename T>
//    void save(const std::string& key, std::vector<T>& obj)
//    {
//        if(_writerPtr)
//        {
//            _writerPtr->Key(key.c_str());

//            save(obj);
//        }
//    }

//    template<typename T, size_t N>
//    void save(const std::string& key, std::array<T, N>& obj)
//    {
//        if(_writerPtr)
//        {
//            _writerPtr->Key(key.c_str());

//            save(obj);
//        }
//    }

    template<typename T>
    void save(const T& obj)
    {
        if(_writerPtr)
        {
            _writerPtr->StartObject();
            obj.save(*this);
            _writerPtr->EndObject();
        }
    }

    template<typename T>
    void save(const std::vector<T>& vector)
    {
        if(_writerPtr)
        {
            _writerPtr->StartArray();
            for(const T& obj : vector)
            {
                _writerPtr->StartObject();
                obj.save(*this);
                _writerPtr->EndObject();
            }
            _writerPtr->EndArray();
        }
    }

    template<typename T, size_t N>
    void save(const std::array<T, N>& array)
    {
        if(_writerPtr)
        {
            _writerPtr->StartArray();
            for(const T& obj : array)
            {
                _writerPtr->StartObject();
                obj.save(*this);
                _writerPtr->EndObject();
            }
            _writerPtr->EndArray();
        }
    }

    void save(bool value)
    {
        if(_writerPtr)
        {
            _writerPtr->Bool(value);
        }
    }

    void save(short int value)
    {
        if(_writerPtr)
        {
            _writerPtr->Int(value);
        }
    }

    void save(unsigned short int value)
    {
        if(_writerPtr)
        {
            _writerPtr->Uint(value);
        }
    }

    void save(int value)
    {
        if(_writerPtr)
        {
            _writerPtr->Int(value);
        }
    }

    void save(unsigned int value)
    {
        if(_writerPtr)
        {
            _writerPtr->Uint(value);
        }
    }

    void save(long int value)
    {
        if(_writerPtr)
        {
            _writerPtr->Int64(value);
        }
    }

    void save(unsigned long int value)
    {
        if(_writerPtr)
        {
            _writerPtr->Uint64(value);
        }
    }

    void save(float value)
    {
        if(_writerPtr)
        {
            _writerPtr->Double(value);
        }
    }

    void save(double value)
    {
        if(_writerPtr)
        {
            _writerPtr->Double(value);
        }
    }

    template<typename T>
    bool write(const T& obj)
    {
        bool success = false;
        std::ofstream ofs(_filename);
        if(ofs)
        {
            rapidjson::OStreamWrapper osw(ofs);
            _writerPtr = std::make_unique<rapidjson::Writer<rapidjson::OStreamWrapper>>(osw);
            _writerPtr->StartObject();
            obj.save(*this);
            _writerPtr->EndObject();
        }

        return success;
    }

    template<typename T>
    bool read(T& obj)
    {
        bool success = false;
        std::ifstream ifs(_filename);
        if(ifs)
        {
            rapidjson::IStreamWrapper isw(ifs);
            _doc.ParseStream(isw);
            _currentNodePtr = &_doc;
            obj.load(*this);
            success = true;
        }

        return success;
    }

    template<typename T>
    void load(T& obj)
    {
        obj.load(*this);
    }

    void load(std::string& value)
    {
        value = _currentNodePtr->GetString();
    }

    void load(bool& value)
    {
        if(_currentNodePtr->IsBool())
        {
            value = _currentNodePtr->GetBool();
        }
        else
        {
            throw std::logic_error("Expected bool");
        }
    }

    void load(int& value)
    {
        if(_currentNodePtr->IsInt())
        {
            value = _currentNodePtr->GetInt();
        }
        else
        {
            throw std::logic_error("Expected int");
        }
    }

    void load(unsigned int& value)
    {
        if(_currentNodePtr->IsInt())
        {
            value = _currentNodePtr->GetInt();
        }
        else
        {
            throw std::logic_error("Expected int");
        }
    }

    void load(long int& value)
    {
        if(_currentNodePtr->IsInt64())
        {
            value = _currentNodePtr->GetInt64();
        }
        else
        {
            throw std::logic_error("Expected int64");
        }
    }

    void load(unsigned long int& value)
    {
        if(_currentNodePtr->IsInt64())
        {
            value = _currentNodePtr->GetInt64();
        }
        else
        {
            throw std::logic_error("Expected int64");
        }
    }

    void load(short int& value)
    {
        if(_currentNodePtr->IsInt())
        {
            value = _currentNodePtr->GetInt();
        }
        else
        {
            throw std::logic_error("Expected int");
        }
    }

    void load(unsigned short int& value)
    {
        if(_currentNodePtr->IsInt())
        {
            value = _currentNodePtr->GetInt();
        }
        else
        {
            throw std::logic_error("Expected int");
        }
    }

    void load(float& value)
    {
        if(_currentNodePtr->IsFloat())
        {
            value = _currentNodePtr->GetFloat();
        }
        else
        {
            throw std::logic_error("Expected float");
        }
    }

    void load(double& value)
    {
        if(_currentNodePtr->IsDouble())
        {
            value = _currentNodePtr->GetDouble();
        }
        else
        {
            throw std::logic_error("Expected double");
        }
    }

    template <typename T>
    void load(const std::string& key, T& obj)
    {
        ValueRestorer holder(_currentNodePtr);
        rapidjson::Value::ConstMemberIterator iter = _currentNodePtr->FindMember(key.c_str());
        if (_currentNodePtr->MemberEnd() == iter)
        {
            throw std::logic_error("No such node (" + key + ")");
        }

        _currentNodePtr = &iter->value;
        load(obj);
    }

    template <typename T>
    void load(std::vector<T>& vector)
    {
        ValueRestorer holder(_currentNodePtr);
        for(rapidjson::Value::ConstValueIterator treeValue = holder.getInitialValue()->Begin();
            treeValue != holder.getInitialValue()->End(); ++treeValue)
        {
            _currentNodePtr = treeValue;
            T t;
            load(t);
            vector.emplace_back(t);
        }
    }

    template <typename T, size_t N>
    void load(std::array<T, N>& array)
    {
        ValueRestorer holder(_currentNodePtr);
        size_t index = 0;
        for(rapidjson::Value::ConstValueIterator treeValue = holder.getInitialValue()->Begin();
            treeValue != holder.getInitialValue()->End(); ++treeValue)
        {
            _currentNodePtr = treeValue;
            T t;
            load(t);
            array[index++] = t;
        }
        if(index != N)
        {
            throw std::logic_error("Array has more elements than present in json");
        }
    }







private:
    std::string _filename;

    const rapidjson::Value *_currentNodePtr;
    rapidjson::Document _doc;
    std::unique_ptr<rapidjson::Writer<rapidjson::OStreamWrapper>> _writerPtr;
};
