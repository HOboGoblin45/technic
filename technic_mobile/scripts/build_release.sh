#!/bin/bash

# ============================================================================
# Technic iOS Release Build Script
# ============================================================================
#
# This script automates the release build process for iOS.
# Run from the technic_mobile directory.
#
# Usage: ./scripts/build_release.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "pubspec.yaml" ]; then
        print_error "pubspec.yaml not found. Please run from technic_mobile directory."
        exit 1
    fi
    print_success "Running from correct directory"
}

# Check Flutter is installed
check_flutter() {
    if ! command -v flutter &> /dev/null; then
        print_error "Flutter is not installed or not in PATH"
        exit 1
    fi
    print_success "Flutter found: $(flutter --version | head -n 1)"
}

# Get current version from pubspec.yaml
get_version() {
    VERSION=$(grep "^version:" pubspec.yaml | sed 's/version: //')
    echo "$VERSION"
}

# Clean build artifacts
clean_build() {
    print_status "Cleaning build artifacts..."
    flutter clean
    print_success "Flutter clean completed"
}

# Get dependencies
get_dependencies() {
    print_status "Getting dependencies..."
    flutter pub get
    print_success "Dependencies retrieved"
}

# Clean and reinstall iOS pods
clean_pods() {
    print_status "Cleaning iOS pods..."
    cd ios
    rm -rf Pods Podfile.lock
    pod install --repo-update
    cd ..
    print_success "Pods reinstalled"
}

# Run Flutter analyze
run_analyze() {
    print_status "Running static analysis..."
    if flutter analyze --no-fatal-infos; then
        print_success "Analysis passed"
    else
        print_warning "Analysis found issues (non-fatal)"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    if flutter test; then
        print_success "All tests passed"
    else
        print_error "Tests failed!"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Build iOS release
build_ios() {
    print_status "Building iOS release..."
    flutter build ios --release
    print_success "iOS release build completed"
}

# Open Xcode
open_xcode() {
    print_status "Opening Xcode..."
    open ios/Runner.xcworkspace
    print_success "Xcode opened"
}

# Main build process
main() {
    echo ""
    echo "=============================================="
    echo "  Technic iOS Release Build"
    echo "=============================================="
    echo ""

    # Run checks
    check_directory
    check_flutter

    # Get current version
    VERSION=$(get_version)
    print_status "Building version: $VERSION"
    echo ""

    # Confirm before proceeding
    read -p "Proceed with release build? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Build cancelled"
        exit 0
    fi

    echo ""
    print_status "Starting build process..."
    echo ""

    # Build steps
    clean_build
    get_dependencies

    # Ask about pod reinstall
    read -p "Reinstall CocoaPods? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        clean_pods
    fi

    # Ask about running tests
    read -p "Run tests before build? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi

    # Ask about analysis
    read -p "Run static analysis? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_analyze
    fi

    # Build iOS
    build_ios

    echo ""
    echo "=============================================="
    print_success "Build completed successfully!"
    echo "=============================================="
    echo ""
    print_status "Next steps:"
    echo "  1. Open Xcode with: open ios/Runner.xcworkspace"
    echo "  2. Select 'Any iOS Device' as destination"
    echo "  3. Go to Product > Archive"
    echo "  4. Validate and upload to App Store Connect"
    echo ""

    # Ask to open Xcode
    read -p "Open Xcode now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open_xcode
    fi

    echo ""
    print_success "Done!"
}

# Run main function
main "$@"
