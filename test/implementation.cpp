#include <catch2/catch.hpp>

#include "core/node.hpp"

namespace Operon::Test {
TEST_CASE("Node is trivial")
{
    REQUIRE(std::is_trivial_v<Operon::Node>);
}
} // namespace Operon::Test
