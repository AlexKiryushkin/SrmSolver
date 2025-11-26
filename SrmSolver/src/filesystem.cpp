
#include "filesystem.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace kae {

std::wstring current_path()
{
  return fs::current_path().wstring();
}

void current_path(const std::wstring & path)
{
  fs::current_path(path);
}

bool create_directories(const std::wstring & path)
{
  return fs::create_directories(path);
}


std::wstring append(const std::wstring & path, const std::wstring & source)
{
  return fs::path{ path }.append(source);
}

std::size_t remove_all(const std::wstring & path)
{
  return static_cast<std::size_t>(fs::remove_all(path));
}

} // namespace kae
