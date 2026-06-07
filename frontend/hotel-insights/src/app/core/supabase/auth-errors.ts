/**
 * Translates Supabase auth errors into user-facing Spanish messages.
 *
 * Matches first by the stable error `code`, then falls back to known English
 * message fragments (OAuth / older SDK errors may not carry a code), and
 * finally returns the original message so nothing is ever swallowed.
 */

export interface AuthErrorLike {
  message?: string;
  code?: string;
}

const CODE_MESSAGES: Record<string, string> = {
  invalid_credentials: 'Correo o contraseña incorrectos.',
  email_not_confirmed: 'Debes confirmar tu correo antes de iniciar sesión.',
  user_already_exists: 'Ya existe una cuenta con este correo.',
  email_exists: 'Ya existe una cuenta con este correo.',
  user_not_found: 'No existe una cuenta con este correo.',
  weak_password: 'La contraseña es demasiado débil. Usa al menos 6 caracteres.',
  email_address_invalid: 'El correo electrónico no es válido.',
  validation_failed: 'Revisa los datos ingresados.',
  signup_disabled: 'El registro está deshabilitado por el momento.',
  same_password: 'La nueva contraseña debe ser diferente a la anterior.',
  session_expired: 'Tu sesión ha expirado. Inicia sesión de nuevo.',
  over_request_rate_limit: 'Demasiados intentos. Inténtalo de nuevo en unos minutos.',
  over_email_send_rate_limit: 'Demasiados intentos. Inténtalo de nuevo en unos minutos.',
};

// Fallback matchers for errors without a code (substring, case-insensitive).
const MESSAGE_MATCHERS: Array<[RegExp, string]> = [
  [/invalid login credentials/i, 'Correo o contraseña incorrectos.'],
  [/email not confirmed/i, 'Debes confirmar tu correo antes de iniciar sesión.'],
  [/user already registered/i, 'Ya existe una cuenta con este correo.'],
  [/password should be at least/i, 'La contraseña debe tener al menos 6 caracteres.'],
  [/unable to validate email address/i, 'El correo electrónico no es válido.'],
  [/for security purposes.*you can only request/i, 'Demasiados intentos. Inténtalo de nuevo en unos minutos.'],
  [/rate limit/i, 'Demasiados intentos. Inténtalo de nuevo en unos minutos.'],
  [/network|failed to fetch/i, 'Error de conexión. Revisa tu internet e inténtalo de nuevo.'],
];

export function translateAuthError(error: AuthErrorLike | null | undefined): string {
  if (!error) return 'Ocurrió un error inesperado.';

  if (error.code && CODE_MESSAGES[error.code]) {
    return CODE_MESSAGES[error.code];
  }

  const message = error.message ?? '';
  for (const [pattern, translated] of MESSAGE_MATCHERS) {
    if (pattern.test(message)) return translated;
  }

  return message || 'Ocurrió un error inesperado.';
}
