"""
Test script for SerperScrapeToolMaz.

This script tests the tool both directly and with a CrewAI agent to ensure
proper functionality for web scraping.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from src.serper_scrape_tool_maz.tool import SerperScrapeToolMaz


def test_tool_directly():
    """Test the tool directly without an agent."""
    print("\n" + "=" * 80)
    print("TEST 1: Direct Tool Testing")
    print("=" * 80 + "\n")

    # Load environment variables
    load_dotenv()

    # Check if API key is available
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("❌ ERROR: SERPER_API_KEY not found in environment variables.")
        print("   Please add it to your .env file.")
        return False

    print(f"✓ API Key found: {api_key[:10]}...")

    # Initialize the tool
    tool = SerperScrapeToolMaz()

    # Test scraping a simple website
    test_url = "https://www.example.com"
    print(f"\n📄 Testing direct scraping of: {test_url}")

    result = tool._run(url=test_url, include_markdown=True)

    if "Error" in result:
        print(f"❌ Tool execution failed:\n{result}")
        return False

    print(f"✓ Successfully scraped content")
    print(f"  Content preview (first 200 chars):")
    print(f"  {result[:200]}...")

    return True


def test_tool_with_agent():
    """Test the tool with a CrewAI agent."""
    print("\n" + "=" * 80)
    print("TEST 2: Agent-Based Tool Testing")
    print("=" * 80 + "\n")

    # Load environment variables
    load_dotenv()

    # Initialize the tool
    scraper_tool = SerperScrapeToolMaz()

    # Create a research agent with the scraper tool
    research_agent = Agent(
        role="Web Content Researcher",
        goal="Extract and analyze web content from URLs to gather information",
        backstory=(
            "You are an expert at gathering information from the web. "
            "You use web scraping tools to extract content from websites "
            "and provide clear, concise summaries of what you find."
        ),
        tools=[scraper_tool],
        verbose=True,
        allow_delegation=False,
    )

    print("✓ Agent created successfully")

    # Create a task for the agent
    scraping_task = Task(
        description=(
            "Use the Serper Web Scraper tool to scrape the content from "
            "https://www.example.com and provide a brief summary of what "
            "you find. Include the main heading and key information."
        ),
        expected_output=(
            "A summary of the scraped webpage content, including the main "
            "heading and key information found on the page."
        ),
        agent=research_agent,
    )

    print("✓ Task created successfully")

    # Create a minimal crew
    crew = Crew(
        agents=[research_agent],
        tasks=[scraping_task],
        verbose=True,
    )

    print("✓ Crew assembled successfully")

    # Execute the crew
    print("\n🚀 Starting crew execution...\n")
    print("-" * 80)

    try:
        result = crew.kickoff()
        print("-" * 80)
        print("\n✓ Crew execution completed successfully")
        print("\n📊 RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Crew execution failed with error: {str(e)}")
        return False


def test_tool_with_multiple_urls():
    """Test the tool with multiple URLs via an agent."""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple URL Scraping Test")
    print("=" * 80 + "\n")

    # Load environment variables
    load_dotenv()

    # Initialize the tool
    scraper_tool = SerperScrapeToolMaz()

    # Create a research agent
    multi_scraper_agent = Agent(
        role="Multi-Site Content Analyzer",
        goal="Scrape multiple websites and compare their content",
        backstory=(
            "You are skilled at gathering information from multiple sources "
            "and synthesizing insights from different websites."
        ),
        tools=[scraper_tool],
        verbose=True,
        allow_delegation=False,
    )

    print("✓ Agent created successfully")

    # Create a task for scraping multiple URLs
    multi_scraping_task = Task(
        description=(
            "Use the Serper Web Scraper tool to scrape content from these URLs:\n"
            "1. https://www.example.com\n"
            "2. https://www.iana.org\n\n"
            "For each URL, provide a brief summary of the main content you find. "
            "Then, provide an overall summary comparing what you learned from both sites."
        ),
        expected_output=(
            "Individual summaries for each URL scraped, followed by a "
            "comparative analysis of the content from both websites."
        ),
        agent=multi_scraper_agent,
    )

    print("✓ Task created successfully")

    # Create a crew
    crew = Crew(
        agents=[multi_scraper_agent],
        tasks=[multi_scraping_task],
        verbose=True,
    )

    print("✓ Crew assembled successfully")

    # Execute the crew
    print("\n🚀 Starting crew execution...\n")
    print("-" * 80)

    try:
        result = crew.kickoff()
        print("-" * 80)
        print("\n✓ Crew execution completed successfully")
        print("\n📊 RESULT:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Crew execution failed with error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🔧" * 40)
    print("SERPER SCRAPE TOOL - COMPREHENSIVE TEST SUITE")
    print("🔧" * 40)

    results = []

    # Test 1: Direct tool usage
    results.append(("Direct Tool Test", test_tool_directly()))

    # Test 2: Agent-based tool usage
    results.append(("Agent-Based Test", test_tool_with_agent()))

    # Test 3: Multiple URLs test
    results.append(("Multiple URLs Test", test_tool_with_multiple_urls()))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("-" * 80)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("=" * 80 + "\n")

    # Return exit code
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    exit(main())

