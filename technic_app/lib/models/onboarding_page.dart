/// Onboarding Page Model
/// 
/// Represents a single page in the onboarding flow.
library;

class OnboardingPage {
  final String title;
  final String description;
  final String imagePath;
  final String? buttonText;

  const OnboardingPage({
    required this.title,
    required this.description,
    required this.imagePath,
    this.buttonText,
  });

  /// Default onboarding pages
  static List<OnboardingPage> get defaultPages => [
        const OnboardingPage(
          title: 'Welcome to Technic',
          description: 'Your AI-powered stock scanner that finds the best trading opportunities in seconds.',
          imagePath: 'assets/onboarding/welcome.png',
        ),
        const OnboardingPage(
          title: 'Smart Scanning',
          description: 'Our advanced algorithms analyze thousands of stocks to find high-probability setups.',
          imagePath: 'assets/onboarding/scanning.png',
        ),
        const OnboardingPage(
          title: 'MERIT Score',
          description: 'Every stock gets a MERIT score based on momentum, earnings, risk, indicators, and technicals.',
          imagePath: 'assets/onboarding/merit.png',
        ),
        const OnboardingPage(
          title: 'Build Your Watchlist',
          description: 'Save your favorite stocks, add notes, set alerts, and track your opportunities.',
          imagePath: 'assets/onboarding/watchlist.png',
        ),
        const OnboardingPage(
          title: 'Ready to Start?',
          description: 'Let\'s find your next winning trade. Tap below to begin scanning!',
          imagePath: 'assets/onboarding/ready.png',
          buttonText: 'Get Started',
        ),
      ];
}
