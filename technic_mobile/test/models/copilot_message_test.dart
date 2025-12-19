import 'package:flutter_test/flutter_test.dart';
import 'package:technic_mobile/models/copilot_message.dart';

void main() {
  group('CopilotMessage', () {
    group('constructor', () {
      test('creates user message', () {
        const message = CopilotMessage(
          'user',
          'What stocks should I look at today?',
        );

        expect(message.role, 'user');
        expect(message.body, 'What stocks should I look at today?');
        expect(message.meta, isNull);
      });

      test('creates assistant message', () {
        const message = CopilotMessage(
          'assistant',
          'Based on current market conditions, consider looking at tech stocks.',
        );

        expect(message.role, 'assistant');
        expect(message.body, 'Based on current market conditions, consider looking at tech stocks.');
      });

      test('creates message with metadata', () {
        const message = CopilotMessage(
          'assistant',
          'AAPL is showing bullish momentum.',
          meta: 'source:scanner',
        );

        expect(message.role, 'assistant');
        expect(message.body, 'AAPL is showing bullish momentum.');
        expect(message.meta, 'source:scanner');
      });
    });

    group('fromJson', () {
      test('parses user message correctly', () {
        final json = {
          'role': 'user',
          'body': 'Tell me about TSLA',
        };

        final message = CopilotMessage.fromJson(json);

        expect(message.role, 'user');
        expect(message.body, 'Tell me about TSLA');
        expect(message.meta, isNull);
      });

      test('parses assistant message correctly', () {
        final json = {
          'role': 'assistant',
          'body': 'TSLA is an electric vehicle manufacturer...',
          'meta': 'ticker:TSLA',
        };

        final message = CopilotMessage.fromJson(json);

        expect(message.role, 'assistant');
        expect(message.body, 'TSLA is an electric vehicle manufacturer...');
        expect(message.meta, 'ticker:TSLA');
      });

      test('handles alternate message key', () {
        final json = {
          'role': 'assistant',
          'message': 'This is the message content',
        };

        final message = CopilotMessage.fromJson(json);

        expect(message.body, 'This is the message content');
      });

      test('prefers body over message when both present', () {
        final json = {
          'role': 'assistant',
          'body': 'From body',
          'message': 'From message',
        };

        final message = CopilotMessage.fromJson(json);

        expect(message.body, 'From body');
      });

      test('handles missing role with default', () {
        final json = {
          'body': 'Some message',
        };

        final message = CopilotMessage.fromJson(json);

        expect(message.role, 'assistant');
      });

      test('handles empty json', () {
        final json = <String, dynamic>{};

        final message = CopilotMessage.fromJson(json);

        expect(message.role, 'assistant');
        expect(message.body, '');
        expect(message.meta, isNull);
      });

      test('handles null values', () {
        final json = {
          'role': null,
          'body': null,
          'meta': null,
        };

        final message = CopilotMessage.fromJson(json);

        expect(message.role, 'assistant');
        expect(message.body, '');
        expect(message.meta, isNull);
      });
    });

    group('toJson', () {
      test('serializes all fields correctly', () {
        const message = CopilotMessage(
          'user',
          'What is the market doing?',
          meta: 'context:general',
        );

        final json = message.toJson();

        expect(json['role'], 'user');
        expect(json['body'], 'What is the market doing?');
        expect(json['meta'], 'context:general');
      });

      test('serializes message without meta', () {
        const message = CopilotMessage(
          'assistant',
          'The market is up today.',
        );

        final json = message.toJson();

        expect(json['role'], 'assistant');
        expect(json['body'], 'The market is up today.');
        expect(json['meta'], isNull);
      });

      test('roundtrip serialization preserves data', () {
        const original = CopilotMessage(
          'user',
          'Analyze NVDA',
          meta: 'request:analysis',
        );

        final json = original.toJson();
        final restored = CopilotMessage.fromJson(json);

        expect(restored.role, original.role);
        expect(restored.body, original.body);
        expect(restored.meta, original.meta);
      });
    });

    group('isUser', () {
      test('returns true for user role', () {
        const message = CopilotMessage('user', 'Hello');

        expect(message.isUser, true);
        expect(message.isAssistant, false);
      });

      test('returns false for assistant role', () {
        const message = CopilotMessage('assistant', 'Hi there');

        expect(message.isUser, false);
      });

      test('returns false for other roles', () {
        const message = CopilotMessage('system', 'System message');

        expect(message.isUser, false);
        expect(message.isAssistant, false);
      });
    });

    group('isAssistant', () {
      test('returns true for assistant role', () {
        const message = CopilotMessage('assistant', 'How can I help?');

        expect(message.isAssistant, true);
        expect(message.isUser, false);
      });

      test('returns false for user role', () {
        const message = CopilotMessage('user', 'Help me');

        expect(message.isAssistant, false);
      });
    });

    group('hasMeta', () {
      test('returns true when meta is set', () {
        const message = CopilotMessage(
          'assistant',
          'Response',
          meta: 'some metadata',
        );

        expect(message.hasMeta, true);
      });

      test('returns false when meta is null', () {
        const message = CopilotMessage('assistant', 'Response');

        expect(message.hasMeta, false);
      });

      test('returns false when meta is empty', () {
        const message = CopilotMessage(
          'assistant',
          'Response',
          meta: '',
        );

        expect(message.hasMeta, false);
      });
    });

    group('edge cases', () {
      test('handles very long message body', () {
        final longBody = 'A' * 10000;
        final message = CopilotMessage('user', longBody);

        expect(message.body.length, 10000);
        expect(message.body, longBody);
      });

      test('handles special characters in body', () {
        const specialChars = r'Hello! @#$%^&*() "quotes" `backticks` \backslash';
        const message = CopilotMessage('user', specialChars);

        expect(message.body, specialChars);
      });

      test('handles unicode characters', () {
        const unicode = 'Hello 你好 مرحبا 🚀 📈 💰';
        const message = CopilotMessage('user', unicode);

        expect(message.body, unicode);
      });

      test('handles newlines in body', () {
        const multiline = 'Line 1\nLine 2\nLine 3';
        const message = CopilotMessage('assistant', multiline);

        expect(message.body, multiline);
        expect(message.body.split('\n').length, 3);
      });
    });
  });
}
