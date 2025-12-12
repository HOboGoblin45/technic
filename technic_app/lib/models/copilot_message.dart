/// CopilotMessage Model
/// 
/// Represents a message in the Copilot chat interface.

class CopilotMessage {
  final String role;
  final String body;
  final String? meta;

  const CopilotMessage(
    this.role,
    this.body, {
    this.meta,
  });

  factory CopilotMessage.fromJson(Map<String, dynamic> json) {
    return CopilotMessage(
      json['role']?.toString() ?? 'assistant',
      json['body']?.toString() ?? json['message']?.toString() ?? '',
      meta: json['meta']?.toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'role': role,
        'body': body,
        'meta': meta,
      };
  
  /// Check if this is a user message
  bool get isUser => role == 'user';
  
  /// Check if this is an assistant message
  bool get isAssistant => role == 'assistant';
  
  /// Check if message has metadata
  bool get hasMeta => meta != null && meta!.isNotEmpty;
}
