/// SavedScreen Model
/// 
/// Represents a saved scanner preset with filters and configuration.
library;

/// A saved scanner configuration preset
class SavedScreen {
  final String name;
  final String description;
  final String horizon;
  final bool isActive;
  final Map<String, String>? params;

  const SavedScreen(
    this.name,
    this.description,
    this.horizon,
    this.isActive, {
    this.params,
  });

  /// Create from JSON
  factory SavedScreen.fromJson(Map<String, dynamic> json) => SavedScreen(
        json['name']?.toString() ?? '',
        json['description']?.toString() ?? '',
        json['horizon']?.toString() ?? '',
        json['isActive'] == true,
        params: (json['params'] as Map?)?.map(
              (k, v) => MapEntry(k.toString(), v.toString()),
            ) ??
            const {},
      );

  /// Convert to JSON
  Map<String, dynamic> toJson() => {
        'name': name,
        'description': description,
        'horizon': horizon,
        'isActive': isActive,
        'params': params,
      };

  /// Create a copy with modified fields
  SavedScreen copyWith({
    String? name,
    String? description,
    String? horizon,
    bool? isActive,
    Map<String, String>? params,
  }) {
    return SavedScreen(
      name ?? this.name,
      description ?? this.description,
      horizon ?? this.horizon,
      isActive ?? this.isActive,
      params: params ?? this.params,
    );
  }
}
