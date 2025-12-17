/// Sort and Filter Bar Widget
/// 
/// Provides sorting and filtering options for scan results.
library;

import 'package:flutter/material.dart';
import '../../../theme/app_colors.dart';
import '../../../utils/helpers.dart';

enum SortOption {
  meritScore('MERIT Score', Icons.verified),
  techRating('Tech Rating', Icons.analytics),
  ics('ICS Score', Icons.business),
  winProb('Win Probability', Icons.percent),
  ticker('Ticker (A-Z)', Icons.sort_by_alpha);

  final String label;
  final IconData icon;
  const SortOption(this.label, this.icon);
}

enum FilterOption {
  all('All Results'),
  highQuality('High Quality (ICS 80+)'),
  core('Core Tier'),
  satellite('Satellite Tier'),
  hasOptions('Has Options');

  final String label;
  const FilterOption(this.label);
}

class SortFilterBar extends StatelessWidget {
  final SortOption currentSort;
  final bool sortDescending;
  final FilterOption currentFilter;
  final int totalResults;
  final int filteredResults;
  final ValueChanged<SortOption> onSortChanged;
  final ValueChanged<bool> onSortDirectionChanged;
  final ValueChanged<FilterOption> onFilterChanged;

  const SortFilterBar({
    super.key,
    required this.currentSort,
    required this.sortDescending,
    required this.currentFilter,
    required this.totalResults,
    required this.filteredResults,
    required this.onSortChanged,
    required this.onSortDirectionChanged,
    required this.onFilterChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: tone(Colors.white, 0.03),
        border: Border(
          bottom: BorderSide(color: tone(Colors.white, 0.08)),
        ),
      ),
      child: Column(
        children: [
          // Sort Row
          Row(
            children: [
              Icon(
                Icons.sort,
                size: 18,
                color: AppColors.primaryBlue,
              ),
              const SizedBox(width: 8),
              const Text(
                'Sort by:',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: SortOption.values.map((option) {
                      final isSelected = currentSort == option;
                      return Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: FilterChip(
                          selected: isSelected,
                          label: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                option.icon,
                                size: 14,
                                color: isSelected
                                    ? Colors.white
                                    : Colors.white60,
                              ),
                              const SizedBox(width: 4),
                              Text(option.label),
                            ],
                          ),
                          onSelected: (_) => onSortChanged(option),
                          backgroundColor: tone(Colors.white, 0.05),
                          selectedColor: AppColors.primaryBlue,
                          checkmarkColor: Colors.white,
                          labelStyle: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: isSelected ? Colors.white : Colors.white60,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ),
              IconButton(
                onPressed: () => onSortDirectionChanged(!sortDescending),
                icon: Icon(
                  sortDescending
                      ? Icons.arrow_downward
                      : Icons.arrow_upward,
                  size: 20,
                ),
                tooltip: sortDescending ? 'Descending' : 'Ascending',
                color: AppColors.primaryBlue,
              ),
            ],
          ),
          
          const SizedBox(height: 12),
          
          // Filter Row
          Row(
            children: [
              Icon(
                Icons.filter_list,
                size: 18,
                color: AppColors.primaryBlue,
              ),
              const SizedBox(width: 8),
              const Text(
                'Filter:',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: FilterOption.values.map((option) {
                      final isSelected = currentFilter == option;
                      return Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: FilterChip(
                          selected: isSelected,
                          label: Text(option.label),
                          onSelected: (_) => onFilterChanged(option),
                          backgroundColor: tone(Colors.white, 0.05),
                          selectedColor: AppColors.primaryBlue,
                          checkmarkColor: Colors.white,
                          labelStyle: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: isSelected ? Colors.white : Colors.white60,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ),
            ],
          ),
          
          // Results Count
          if (filteredResults != totalResults) ...[
            const SizedBox(height: 8),
            Text(
              'Showing $filteredResults of $totalResults results',
              style: const TextStyle(
                fontSize: 12,
                color: Colors.white38,
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
