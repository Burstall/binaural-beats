// A very small IndexedDB wrapper: one database, one "outbox" store of API
// writes made while offline, sent in order when the app is next online.

const DB_NAME = "violet";
const STORE = "outbox";

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "seq", autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const result = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

/** Queues a write, replacing any queued write to the same method and path. */
export async function enqueue(entry) {
  const all = await list();
  await tx("readwrite", (store) => {
    for (const old of all) if (old.method === entry.method && old.path === entry.path) store.delete(old.seq);
    store.add({ ...entry, at: Date.now() });
  });
}

export async function list() {
  return tx("readonly", (store) => store.getAll());
}

export async function remove(seq) {
  await tx("readwrite", (store) => store.delete(seq));
}
