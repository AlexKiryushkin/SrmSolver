#pragma once

#include <string>

namespace kae {

std::wstring current_path();
void current_path(const std::wstring & path);

bool create_directories(const std::wstring & path);

std::wstring append(const std::wstring & path, const std::wstring & source);

std::size_t remove_all(const std::wstring & path);

} // namespace kae
