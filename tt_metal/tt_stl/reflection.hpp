/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <fmt/core.h>

#include <boost/core/demangle.hpp>
#include <experimental/type_traits>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "third_party/magic_enum/magic_enum.hpp"

namespace tt {
namespace stl {

// Forward Declare hash_object
namespace hash {

constexpr bool DEBUG_HASH_OBJECT_FUNCTION = false;

using hash_t = std::uint64_t;

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>& variant) noexcept;

template <typename T>
inline hash_t hash_object(const T& object) noexcept;

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... objects) noexcept;

}  // namespace hash

namespace reflection {

using AttributeName = std::variant<const char*, std::string>;

struct Attribute {
    const std::function<std::string()> to_string;
    const std::function<hash::hash_t()> to_hash;

    template <typename T>
    Attribute(const T& value) :
        to_string([value] { return fmt::format("{}", value); }),
        to_hash([value] { return hash::hash_object(value); }) {}
};

using Attributes = std::vector<std::tuple<AttributeName, Attribute>>;



namespace detail {
template <typename T>
using has_attributes_t = decltype(std::declval<T>().attributes());

template <typename T>
constexpr bool supports_runtime_time_attributes_v = std::experimental::is_detected_v<has_attributes_t, T>;

template <typename T>
inline constexpr std::size_t get_num_attributes() {
    return std::tuple_size_v<decltype(T::attribute_names)>;
}
template <typename T>
using has_attribute_names_t = decltype(std::declval<T>().attribute_names);

template <typename T>
using has_attribute_values_t = decltype(std::declval<T>().attribute_values());

template <typename T>
constexpr bool supports_compile_time_attributes_v = std::experimental::is_detected_v<has_attribute_names_t, T> and
                                                    std::experimental::is_detected_v<has_attribute_values_t, T>;
}  // namespace detail

template<typename T>
Attributes get_attributes(const T& object) {
    if constexpr (tt::stl::reflection::detail::supports_compile_time_attributes_v<std::decay_t<T>>) {
        constexpr auto num_attributes = tt::stl::reflection::detail::get_num_attributes<std::decay_t<T>>();
        tt::stl::reflection::Attributes attributes;
        [&object, &attributes]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &attributes] {
                    const auto& attribute_name = std::get<Ns>(object.attribute_names);
                    const auto& attribute = std::get<Ns>(object.attribute_values());
                    attributes.push_back({attribute_name, attribute});
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return attributes;
    } else if constexpr (tt::stl::reflection::detail::supports_runtime_time_attributes_v<std::decay_t<T>>) {
        return object.attributes();
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<T>,
            "Object doesn't support compile-time or run-time attributes!");
    }
}


static std::ostream& operator<<(std::ostream& os, const Attribute& attribute) {
    os << attribute.to_string();
    return os;
}

static std::ostream& operator<<(std::ostream& os, const Attributes& attributes) {
    os << "(";
    for (auto index = 0; index < attributes.size(); index++) {
        auto&& [key, value] = attributes[index];
        os << key << "=" << value;
        if (index != attributes.size() - 1) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

template <typename T>
typename std::enable_if_t<detail::supports_runtime_time_attributes_v<T>, std::ostream>& operator<<(
    std::ostream& os, const T& object) {
    static_assert(std::is_same_v<decltype(object.attributes()), Attributes>);
    os << boost::core::demangle(typeid(T).name());
    os << object.attributes();
    return os;
}

template <typename T>
typename std::enable_if_t<detail::supports_compile_time_attributes_v<T>, std::ostream>& operator<<(
    std::ostream& os, const T& object) {
    constexpr auto num_attributes = detail::get_num_attributes<T>();
    os << boost::core::demangle(typeid(T).name());
    os << "(";

    if constexpr (num_attributes > 0) {
        [&os, &object]<std::size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&os, &object] {
                    const auto& attribute = std::get<Ns>(object.attribute_values());
                    os << std::get<Ns>(object.attribute_names);
                    os << "=";
                    os << attribute;
                    os << ",";
                }(),
                ...);
        }(std::make_index_sequence<num_attributes - 1>{});

        const auto& attribute = std::get<num_attributes - 1>(object.attribute_values());
        os << std::get<num_attributes - 1>(object.attribute_names);
        os << "=";
        os << attribute;
    }

    os << ")";
    return os;
}

template <typename T>
typename std::enable_if_t<std::is_enum<T>::value, std::ostream>& operator<<(std::ostream& os, const T& value) {
    os << magic_enum::enum_type_name<T>() << "::" << magic_enum::enum_name(value);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional) {
    if (optional.has_value()) {
        os << optional.value();
    } else {
        os << "std::nullopt";
    }
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::variant<Ts...>& variant) {
    std::visit([&os](const auto& value) { os << value; }, variant);
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& array) {
    os << "{";
    for (auto index = 0; index < array.size(); index++) {
        const auto& element = array[index];
        os << element;
        if (index != array.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    os << "{";
    for (auto index = 0; index < vector.size(); index++) {
        const auto& element = vector[index];
        os << element;
        if (index != vector.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

}  // namespace reflection
}  // namespace stl
}  // namespace tt

template <>
struct fmt::formatter<tt::stl::reflection::Attributes> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::stl::reflection::Attributes& attributes, format_context& ctx) const
        -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << attributes;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<tt::stl::reflection::detail::supports_compile_time_attributes_v<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<tt::stl::reflection::detail::supports_runtime_time_attributes_v<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<std::is_enum<T>::value>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const T& value, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << value;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T>
struct fmt::formatter<std::optional<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::optional<T>& optional, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << optional;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename ... Ts>
struct fmt::formatter<std::variant<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::variant<Ts...>& variant, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << variant;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T, std::size_t N>
struct fmt::formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::array<T, N>& array, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << array;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};


template <typename T>
struct fmt::formatter<std::vector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator {
        return ctx.end();
    }

    auto format(const std::vector<T>& vector, format_context& ctx) const -> format_context::iterator {
        using tt::stl::reflection::operator<<;
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

namespace tt {
namespace stl {
namespace hash {

namespace detail {
template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template <typename Test, template <typename...> class Ref>
constexpr bool is_specialization_v = is_specialization<Test, Ref>::value;

}  // namespace detail

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::array<{}, {}>\n", boost::core::demangle(typeid(T).name()), N);
    }
    std::size_t hash = 0;
    [&array, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
        (
            [&array, &hash] {
                const auto& element = std::get<Ns>(array);
                hash = hash_objects(hash, element);
            }(),
            ...);
    }(std::make_index_sequence<N>{});
    return hash;
}

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>& variant) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::variant\n");
    }
    return std::visit([](const auto& value) { return hash_object(value); }, variant);
}

template <typename T>
inline hash_t hash_object(const T& object) noexcept {
    if constexpr (std::numeric_limits<T>::is_integer) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing integer of type {}\n", boost::core::demangle(typeid(T).name()));
        }
        return object;
    } else if constexpr (detail::is_std_hashable_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing {} using std::hash\n", boost::core::demangle(typeid(T).name()));
        }
        return std::hash<T>{}(object);
    } else if constexpr (std::is_same_v<T, tt::stl::reflection::Attribute>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing tt::stl::reflection::Attribute\n");
        }
        return object.to_hash();
    } else if constexpr (std::is_same_v<T, tt::stl::reflection::Attributes>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing tt::stl::reflection::Attributes\n");
        }
        auto hash = 0;
        for (auto&& [name, attribute] : object) {
            hash = hash_objects(hash, attribute);
        }
        return hash;
    } else if constexpr (tt::stl::reflection::detail::supports_compile_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using compile-time attributes\n", boost::core::demangle(typeid(T).name()));
        }
        constexpr auto num_attributes = reflection::detail::get_num_attributes<T>();
        std::size_t hash = 0;
        [&object, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &hash] {
                    const auto& attribute = std::get<Ns>(object.attribute_values());
                    hash = hash_objects(hash, attribute);
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return hash;
    } else if constexpr (tt::stl::reflection::detail::supports_runtime_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using run-time attributes\n", boost::core::demangle(typeid(T).name()));
        }
        return hash_object(object.attributes());
    } else if constexpr (detail::is_specialization_v<T, std::vector>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::vector<{}>\n", boost::core::demangle(typeid(T).name()));
        }
        auto hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (detail::is_specialization_v<T, std::optional>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::optional<{}>\n", boost::core::demangle(typeid(T).name()));
        }
        if (object.has_value()) {
            return hash_object(object.value());
        } else {
            return 0;
        }
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Type doesn't support std::hash");
        return 0;
    }
}

namespace detail {

template <typename Type, typename... Types>
inline hash_t hash_objects(hash_t seed, const Type& object, const Types&... objects) noexcept {
    seed = hash_object(object) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    if constexpr (sizeof...(objects) > 0) {
        seed = hash_objects(seed, objects...);
    }
    return seed;
}
}  // namespace detail

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... objects) noexcept {
    return detail::hash_objects(seed, objects...);
}

}  // namespace hash
}  // namespace stl
}  // namespace tt
