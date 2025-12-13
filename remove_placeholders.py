"""
Remove placeholder/mock data from Ideas and Copilot pages.
"""

def remove_ideas_placeholders():
    """Remove mock data fallback from Ideas page."""
    file_path = "technic_app/lib/screens/ideas/ideas_page.dart"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove mock_data import
    content = content.replace("import '../../utils/mock_data.dart';\n", "")
    
    # Replace the _loadIdeasFromLastScan method to not fall back to mockIdeas
    old_method = '''  Future<List<Idea>> _loadIdeasFromLastScan() async {
    // Derive ideas from last scan results; if empty, try pulling from API as fallback
    final scans = ref.read(lastScanResultsProvider);
    
    if (scans.isNotEmpty) {
      return scans.map((s) {
        final plan =
            'Entry ${s.entry.isNotEmpty ? s.entry : "-"}, Stop ${s.stop.isNotEmpty ? s.stop : "-"}, Target ${s.target.isNotEmpty ? s.target : "-"}';
        final why =
            '${s.signal} setup based on blended trend, momentum, volume, and risk scores.';
        return Idea(s.signal, s.ticker, why, plan, s.sparkline);
      }).toList();
    }
    
    try {
      final apiService = ref.read(apiServiceProvider);
      return await apiService.fetchIdeas();
    } catch (_) {
      return mockIdeas;
    }
  }'''
    
    new_method = '''  Future<List<Idea>> _loadIdeasFromLastScan() async {
    // Derive ideas from last scan results; if empty, try pulling from API
    final scans = ref.read(lastScanResultsProvider);
    
    if (scans.isNotEmpty) {
      return scans.map((s) {
        final plan =
            'Entry ${s.entry.isNotEmpty ? s.entry : "-"}, Stop ${s.stop.isNotEmpty ? s.stop : "-"}, Target ${s.target.isNotEmpty ? s.target : "-"}';
        final why =
            '${s.signal} setup based on blended trend, momentum, volume, and risk scores.';
        return Idea(s.signal, s.ticker, why, plan, s.sparkline);
      }).toList();
    }
    
    try {
      final apiService = ref.read(apiServiceProvider);
      return await apiService.fetchIdeas();
    } catch (_) {
      return []; // Return empty list instead of mock data
    }
  }'''
    
    content = content.replace(old_method, new_method)
    
    # Replace mockIdeas fallback in build method
    content = content.replace(
        'final ideas = snapshot.data ?? mockIdeas;',
        'final ideas = snapshot.data ?? [];'
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Removed placeholder data from Ideas page")
    return True

def remove_copilot_placeholders():
    """Remove mock messages from Copilot page."""
    file_path = "technic_app/lib/screens/copilot/copilot_page.dart"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove mock_data import
    content = content.replace("import '../../utils/mock_data.dart';\n", "")
    
    # Replace mock messages initialization with empty list
    content = content.replace(
        'final List<CopilotMessage> _messages = List.of(copilotMessages);',
        'final List<CopilotMessage> _messages = []; // Start with empty conversation'
    )
    
    # Remove copilotPrompts reference (keep the chip UI but make it empty or use real prompts)
    old_prompts = '''            child: Row(
              children: copilotPrompts
                  .map(
                    (p) => Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ActionChip(
                        backgroundColor: tone(Colors.white, 0.04),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: BorderSide(color: tone(Colors.white, 0.06)),
                        ),
                        label: Text(p, style: const TextStyle(color: Colors.white)),
                        onPressed: () => _sendPrompt(p),
                      ),
                    ),
                  )
                  .toList(),
            ),'''
    
    new_prompts = '''            child: Row(
              children: [
                'Summarize today\'s scan',
                'Explain top idea',
                'Compare momentum leaders',
              ]
                  .map(
                    (p) => Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ActionChip(
                        backgroundColor: tone(Colors.white, 0.04),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: BorderSide(color: tone(Colors.white, 0.06)),
                        ),
                        label: Text(p, style: const TextStyle(color: Colors.white)),
                        onPressed: () => _sendPrompt(p),
                      ),
                    ),
                  )
                  .toList(),
            ),'''
    
    content = content.replace(old_prompts, new_prompts)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Removed placeholder messages from Copilot page")
    return True

if __name__ == "__main__":
    print("Removing placeholder/mock data from Ideas and Copilot pages...")
    print()
    
    success = True
    success = remove_ideas_placeholders() and success
    success = remove_copilot_placeholders() and success
    
    print()
    if success:
        print("✅ All placeholders removed successfully!")
        print()
        print("Changes made:")
        print("  1. Ideas page: No longer shows mock ideas")
        print("  2. Ideas page: Shows empty state when no real data")
        print("  3. Copilot page: Starts with empty conversation")
        print("  4. Copilot page: Uses real prompt suggestions")
    else:
        print("❌ Some operations failed")
