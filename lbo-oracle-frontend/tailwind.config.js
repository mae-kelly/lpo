/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#f0f4f8',
          100: '#d9e6f2',
          200: '#b8d1e8',
          300: '#8fb5d9',
          400: '#5f8ec4',
          500: '#3d6bad',
          600: '#2d5494',
          700: '#1e3a6f',
          800: '#14294f',
          900: '#0c1a33',
        },
        gold: {
          50: '#fefcf0',
          100: '#fef7d7',
          200: '#fdecae',
          300: '#fbdc7a',
          400: '#f7c843',
          500: '#d4af37',
          600: '#b8941f',
          700: '#947518',
          800: '#785d17',
          900: '#644918',
        }
      },
      fontFamily: {
        serif: ['Crimson Text', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
