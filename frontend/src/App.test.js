import { render, screen } from '@testing-library/react';
import App from './App';

test('renders app and navigation logo text', () => {
  render(<App />);
  const logoText = screen.getByText(/SAFION/i);
  expect(logoText).toBeInTheDocument();
});
