/// UniverseStats Model
/// 
/// Statistics about the trading universe (sectors, subindustries, total symbols).

class UniverseStats {
  final int total;
  final Map<String, int> sectors;
  final Map<String, int> subindustries;

  const UniverseStats(
    this.total,
    this.sectors,
    this.subindustries,
  );

  factory UniverseStats.fromJson(Map<String, dynamic> json) {
    final total = json['total'] is num ? (json['total'] as num).toInt() : 0;
    
    final sectorsRaw = json['sectors'] as List<dynamic>? ?? [];
    final sectors = <String, int>{};
    for (final s in sectorsRaw) {
      if (s is Map<String, dynamic>) {
        final name = s['name']?.toString() ?? '';
        final count = s['count'] is num ? (s['count'] as num).toInt() : 0;
        if (name.isNotEmpty) sectors[name] = count;
      }
    }
    
    final subsRaw = json['subindustries'] as List<dynamic>? ?? [];
    final subs = <String, int>{};
    for (final s in subsRaw) {
      if (s is Map<String, dynamic>) {
        final name = s['name']?.toString() ?? '';
        final count = s['count'] is num ? (s['count'] as num).toInt() : 0;
        if (name.isNotEmpty) subs[name] = count;
      }
    }
    
    return UniverseStats(total, sectors, subs);
  }

  Map<String, dynamic> toJson() => {
        'total': total,
        'sectors': sectors.entries
            .map((e) => {'name': e.key, 'count': e.value})
            .toList(),
        'subindustries': subindustries.entries
            .map((e) => {'name': e.key, 'count': e.value})
            .toList(),
      };
  
  /// Get total count for selected sectors
  int getSectorCount(List<String> selectedSectors) {
    if (selectedSectors.isEmpty) return total;
    return selectedSectors
        .map((s) => sectors[s] ?? 0)
        .fold<int>(0, (a, b) => a + b);
  }
  
  /// Get total count for selected subindustries
  int getSubindustryCount(List<String> selectedSubindustries) {
    if (selectedSubindustries.isEmpty) return 0;
    return selectedSubindustries
        .map((s) => subindustries[s] ?? 0)
        .fold<int>(0, (a, b) => a + b);
  }
  
  /// Get combined count for selections
  int getCombinedCount(List<String> selectedSectors, List<String> selectedSubindustries) {
    if (selectedSectors.isEmpty && selectedSubindustries.isEmpty) {
      return total;
    }
    final sectorSum = getSectorCount(selectedSectors);
    final subSum = getSubindustryCount(selectedSubindustries);
    final sum = sectorSum + subSum;
    return sum > 0 ? sum : total;
  }
}
