/** The server a source-checkout host starts gets the bundle's UTF-8 settings and no STRATA_API_KEY: the run config carries
 * this instance's key, an inherited one must not override it, and a present-but-empty one is refused (#213, #3851). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { sourceServerEnvironment } from '../server/native-control.mjs';

test('the source-mode server environment keeps the process environment and the UTF-8 settings', () => {
  const env = sourceServerEnvironment({ PATH: '/usr/bin', HOME: '/home/x', KEEP: '1' });
  assert.equal(env.PATH, '/usr/bin');
  assert.equal(env.HOME, '/home/x');
  assert.equal(env.KEEP, '1');
  assert.equal(env.PYTHONUNBUFFERED, '1');
  assert.equal(env.PYTHONUTF8, '1');
  assert.equal(env.PYTHONIOENCODING, 'utf-8');
});

test('an inherited STRATA_API_KEY is removed, not blanked', () => {
  const env = sourceServerEnvironment({ STRATA_API_KEY: 'inherited-from-the-shell' });
  assert.equal('STRATA_API_KEY' in env, false);
});

test('the environment it is given is not modified', () => {
  const base = { STRATA_API_KEY: 'inherited' };
  sourceServerEnvironment(base);
  assert.equal(base.STRATA_API_KEY, 'inherited');
});

test('a process without the variable gets the same result', () => {
  const env = sourceServerEnvironment({});
  assert.equal('STRATA_API_KEY' in env, false);
});
