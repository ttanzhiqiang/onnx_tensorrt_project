#include "common.h"

bool fileExists(const std::string fileName)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        return false;
    }
    return true;
}