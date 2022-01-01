/* Copyright (C) 2010, Armin Biere, JKU Linz. */

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include "uthash.h"
#include <gmp.h>
#include "depqbf/qdpll.h"

extern const char *blqr_id (void);
extern const char *blqr_version (void);
extern const char *blqr_cflags (void);
QDPLL * qdpll;

#define COVER(COVER_CONDITION) \
do { \
  if (!(COVER_CONDITION)) break; \
  fprintf (stderr, \
           "*BLOQQER* covered line %d: %s\n", \
	   __LINE__, # COVER_CONDITION); \
  if (getenv("NCOVER")) break; \
  abort (); \
} while (0)

#ifdef NLOG
#define LOG(...) do { } while (0)
#define LOGCLAUSE(...) do { } while (0)
#else
static int loglevel;
#define LOGPREFIX "c [BLOQQER] "
#define LOG(FMT,ARGS...) \
  do { \
    if (loglevel <= 0) break; \
    fputs (LOGPREFIX, stdout); \
    fprintf (stdout, FMT, ##ARGS); \
    fputc ('\n', stdout); \
    fflush (stdout); \
  } while (0)
#define LOGCLAUSE(CLAUSE,FMT,ARGS...) \
  do { \
    Node * LOGCLAUSE_P; \
    if (loglevel <= 0) break; \
    fputs (LOGPREFIX, stdout); \
    fprintf (stdout, FMT, ##ARGS); \
    for (LOGCLAUSE_P = (CLAUSE)->nodes; LOGCLAUSE_P->lit; LOGCLAUSE_P++) \
      fprintf (stdout, " %d", LOGCLAUSE_P->lit); \
    fputc ('\n', stdout); \
    fflush (stdout); \
  } while (0)
#endif

#define INC(B) \
  do { \
    current_bytes += B; \
    if (max_bytes < current_bytes) \
      max_bytes = current_bytes; \
  } while (0)
#define DEC(B) \
  do { \
    assert (current_bytes >= B); \
    current_bytes -= B; \
  } while (0)
#define NEWN(P,N) \
  do { \
    size_t NEWN_BYTES = (N) * sizeof *(P); \
    (P) = malloc (NEWN_BYTES); \
    if (!(P)) die ("out of memory"); \
    memset ((P), 0, NEWN_BYTES); \
    INC (NEWN_BYTES); \
  } while (0)
#define DELN(P,N) \
  do { \
    size_t DELN_BYTES = (N) * sizeof *(P); \
    DEC (DELN_BYTES); \
    free (P); \
  } while (0)
#define RSZ(P,M,N) \
  do { \
    size_t RSZ_OLD_BYTES = (M) * sizeof *(P); \
    size_t RSZ_NEW_BYTES = (N) * sizeof *(P); \
    DEC (RSZ_OLD_BYTES); \
    (P) = realloc (P, RSZ_NEW_BYTES); \
    if (RSZ_OLD_BYTES > 0 && !(P)) die ("out of memory"); \
    if (RSZ_NEW_BYTES > RSZ_OLD_BYTES) { \
      size_t RSZ_INC_BYTES = RSZ_NEW_BYTES - RSZ_OLD_BYTES; \
      memset (RSZ_OLD_BYTES + ((char*)(P)), 0, RSZ_INC_BYTES); \
    } \
    INC (RSZ_NEW_BYTES); \
  } while (0)
#define NEW(P) NEWN((P),1)
#define DEL(P) DELN((P),1)

typedef struct Occ {		/* occurrence list anchor */
  int count;
  struct Node * first, * last;
} Occ;

typedef struct Var {
  int mapped;
  int scope_idx;
  int free;
  struct Scope * scope;
  struct Var * next;	/* scope variable list links */
  Occ occs[2];			/* positive and negative occurrence lists */
} Var;

typedef struct Scope {
  int type;			/* type>0 == existential, type<0 universal */
  int nr_vars;			/* number of variables in this scope */
  int order;
  struct Var * first, * last;	/* variable list */
  struct Scope * inner;
} Scope;

typedef struct Node {		/* one 'lit' occurrence in a 'clause' */
  int lit;
  struct Clause * clause;
  struct Node * prev, * next;	/* links all occurrences of 'lit' */
} Node;


typedef struct Clause {
  int size;
  int orig_size;
  struct Clause * prev, * next;	/* chronlogical clause list links */
  Node nodes[1];		/* embedded literal nodes */
} Clause;


static int verbose, force;
static FILE * ifile, * ofile;


#define IM INT_MAX

static int remaining_clauses_to_parse;
static int ifclose;
static int lineno = 1;

static int notclean, quiet;

static Scope * inner_most_scope, * outer_most_scope;
static Clause * first_clause, * last_clause;
static int empty_clause = 0;

static int num_vars, universal_vars, existential_vars;
static Var * vars;

static int num_lits, size_lits, * lits;
static int nline, szline;
static char * line;

static size_t current_bytes, max_bytes;
static size_t num_clauses;
static int remaining, mapped, scope_idx;
static int SAT_STATUS;
static int SAT = 10;
static int UNSAT = 20;
static int * forced_stack, forced_lits;
static int size_forced = 8;
static int CORRECT_RESULT = 1;

static void clean_line (void) {
  int i;
  if (!notclean) return;
  fputc ('\r', stdout);
  for (i = 0; i < 70; i++) fputc (' ', stdout);
  fputc ('\r', stdout);
  fflush (stdout);
  notclean = 0;
}

static void die (const char * fmt, ...) {
  va_list ap;
  fputs ("*** bloqqer: ", stderr);
  va_start (ap, fmt);
  vfprintf (stderr, fmt, ap);
  va_end (ap);
  fputc ('\n', stderr);
  fflush (stderr);
  exit (1);
}

static void msg (const char * fmt, ...) {
  va_list ap;
  if (!verbose) return;
  if (notclean) clean_line ();
  fputs ("c [bloqqer] ", stdout);
  va_start (ap, fmt);
  vfprintf (stdout, fmt, ap);
  va_end (ap);
  fputc ('\n', stdout);
  fflush (stdout);
}

static Var * lit2var (int lit) {
  assert (lit && abs (lit) <= num_vars);
  return vars + abs (lit);
}

static Scope * lit2scope (int lit) {
  return lit2var (lit)->scope;
}

static int lit2order (int lit) {
  return lit2scope (lit)->order;
}

static int existential (int lit) {
  return lit2scope (lit)->type > 0;
}

static int universal (int lit) {
  return lit2scope (lit)->type < 0;
}

static int sign (int lit) {
  return lit < 0 ? -1 : 1;
}

static void enlarge_forced (void) {
  int new_size = size_forced ? 2*size_forced : 1;
  RSZ (forced_stack, size_forced, new_size);
  size_forced = new_size;
}

static void push_forced (int lit) {
  if (size_forced == forced_lits) enlarge_forced ();
  forced_stack[forced_lits++] = lit;
}

static int pop_forced_lit () {
    assert (forced_lits > 0);
    return forced_stack[--forced_lits];

}

static void add_outer_most_scope (int lit) {
  NEW (outer_most_scope);
  outer_most_scope->type = sign (lit);
  outer_most_scope->nr_vars = 0;
  inner_most_scope = outer_most_scope;
}

static void add_var (int idx, Scope * scope) {
  Var * v;
  assert (0 < idx && idx <= num_vars);
  v = lit2var (idx);
  assert (!v->scope);
  v->scope = scope;
  v->scope_idx = ++scope_idx;
  v->free = 1;
  if (scope->last) scope->last->next = v;
  else scope->first = v;
  scope->last = v;
  scope->nr_vars++;
}

static void add_quantifier (int lit) {
  Scope * scope;
  if (!outer_most_scope) add_outer_most_scope (sign(lit));
  if (inner_most_scope->type != sign (lit)) {
    NEW (scope);
    scope->type = sign (lit);
    scope->order = inner_most_scope->order + 1;
    inner_most_scope->inner = scope;
    inner_most_scope = scope;
  } else scope = inner_most_scope;
  add_var (abs (lit), scope);
}

static size_t bytes_clause (int size) {
  return sizeof (Clause) + size * sizeof (Node);
}

static Occ * lit2occ (int lit) {
  Var * v = lit2var (lit);
  return v->occs + (lit < 0);
}

static void forall_reduce_clause (void) {
  int i, j, lit, order, tmp;
  order = 0;
  for (i = 0; i < num_lits; i++) {
    lit = lits[i];
    if (universal (lit)) continue;
    tmp = lit2order (lit);
    if (tmp <= order) continue;
    order = tmp;
  }
  j = 0;
  for (i = 0; i < num_lits; i++) {
    lit = lits[i];
    if (existential (lit) || lit2order (lit) < order) lits[j++] = lit;
  }
  num_lits = j;
}


static void add_node (Clause * clause, Node * node, int lit) {
  Occ * occ;

  assert (clause->nodes <= node && node < clause->nodes + clause->size);
  node->clause = clause;
  node->lit = lit;
  occ = lit2occ (lit);
  node->prev = occ->last;
  if (occ->last) occ->last->next = node;
  else occ->first = node;
  occ->last = node;
  occ->count++;
}

static void add_clause (void) {
  Clause * clause;
  size_t bytes;
  int i;
  
  forall_reduce_clause ();
  
  if (!num_lits) {empty_clause = 1; SAT_STATUS = UNSAT; return;}
  
  bytes = bytes_clause (num_lits);
  clause = malloc (bytes);
  if (!clause) die ("out of memory");
  memset (clause, 0, bytes);
  assert (!clause->nodes[num_lits].lit);
  INC (bytes);
  clause->size = num_lits;
  clause->orig_size = num_lits;
  clause->prev = last_clause;
  if (last_clause) last_clause->next = clause;
  else first_clause = clause;
  last_clause = clause;
  for (i = 0; i < num_lits; i++)
    add_node (clause, clause->nodes + i, lits[i]);
  // if forall reduced clause has size 1, the lit must be existential
  if (num_lits == 1) push_forced(lits[0]);
  num_lits = 0;
  num_clauses++;
}

static void enlarge_lits (void) {
  int new_size_lits = size_lits ? 2*size_lits : 1;
  RSZ (lits, size_lits, new_size_lits);
  size_lits = new_size_lits;
}

static void push_literal (int lit) {
  assert (abs (lit) <= num_vars);
  if (size_lits == num_lits) enlarge_lits ();
  lits[num_lits++] = lit;
}


static void init_variables (void) {
  int i;
  assert (1 <= num_vars);
  for (i = 1; i <= num_vars; i++) {
    Var * v = vars + i;
    v->mapped = ++mapped;
    assert (mapped == i);
  }
  assert (mapped == num_vars);
}


static const char * parse (void) {
  int ch, m, n, i, c, q, lit, sign;

  lineno = 1;
  m = n = i = c = q = 0;
  assert (!universal_vars);
  assert (!existential_vars);
  szline = 128;
  assert (!line);
  NEWN (line, szline);
  nline = 0;
SKIP:
  ch = getc (ifile);
  if (ch == '\n') { lineno++; goto SKIP; }
  if (ch == ' ' || ch == '\t' || ch == '\r') goto SKIP;
  if (ch == 'c') {
    line[nline = 0] = 0;
    while ((ch = getc (ifile)) != '\n') {
      if (ch == EOF) return "end of file in comment";
      if (nline + 1 == szline) {
	RSZ (line, szline, 2*szline);
	szline *= 2;
      }
      line[nline++] = ch;
      line[nline] = 0;
    }
    lineno++;
    goto SKIP;
  }
  if (ch != 'p') {
HERR:
    return "invalid or missing header";
  }
  if (getc (ifile) != ' ') goto HERR;
  while ((ch = getc (ifile)) == ' ')
    ;
  if (ch != 'c') goto HERR;
  if (getc (ifile) != 'n') goto HERR;
  if (getc (ifile) != 'f') goto HERR;
  if (getc (ifile) != ' ') goto HERR;
  while ((ch = getc (ifile)) == ' ')
    ;
  if (!isdigit (ch)) goto HERR;
  m = ch - '0';
  while (isdigit (ch = getc (ifile)))
    m = 10 * m + (ch - '0');
  if (ch != ' ') goto HERR;
  while ((ch = getc (ifile)) == ' ')
    ;
  if (!isdigit (ch)) goto HERR;
  n = ch - '0';
  while (isdigit (ch = getc (ifile)))
    n = 10 * n + (ch - '0');
  while (ch != '\n')
    if (ch != ' ' && ch != '\t' && ch != '\r') goto HERR;
    else ch = getc (ifile);
  lineno++;
  msg ("found header 'p cnf %d %d'", m, n);
  remaining = num_vars = m;
  remaining_clauses_to_parse = n;
  NEWN (vars, num_vars + 1);
  if (num_vars) init_variables ();
NEXT:
   ch = getc (ifile);
   if (ch == '\n') { lineno++; goto NEXT; }
   if (ch == ' ' || ch == '\t' || ch == '\r') goto NEXT;
   if (ch == 'c') {
     while ((ch = getc (ifile)) != '\n')
       if (ch == EOF) return "end of file in comment";
     lineno++;
     goto NEXT;
   }
   if (ch == EOF) {
     if (!force && i < n) return "clauses missing";
     goto DONE;
   }
   if (ch == '-') {
     if (q) return "negative number in precix";
     sign = -1;
     ch = getc (ifile);
     if (ch == '0') return "'-' followed by '0'";
   } else sign = 1;
   if (ch == 'e') { 
     if (c) return "'e' after at least one clause";
     if (q) return "'0' missing after 'e'";
     q = 1;
     goto NEXT;
   }
   if (ch == 'a') { 
     if (c) return "'a' after at least one cluase";
     if (q) return "'0' missing after 'a'";
     q = -1; 
     goto NEXT;
   }
   if (!isdigit (ch)) return "expected digit";
   lit = ch - '0';
   while (isdigit (ch = getc (ifile)))
     lit = 10 * lit + (ch - '0');
   if (ch != EOF && ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
     return "expected space after literal";
   if (ch == '\n') lineno++;
   if (lit > m) return "maximum variable index exceeded";
   if (!force && !q && i == n) return "too many clauses";
   if (q) {
     if (lit) {
       if (sign < 0) return "negative literal quantified";
       if (lit2scope (lit)) return "variable quantified twice";
       lit *= q;
       add_quantifier (lit);
       if (q > 0) existential_vars++; else universal_vars++;
     } else q = 0;
   }
   else {
     if (lit) lit *= sign, c++; else i++, remaining_clauses_to_parse--;
     if (lit) push_literal (lit);
     else add_clause ();
   }
   goto NEXT;
DONE:
  return 0;
}

static int var2lit (Var * v) { 
  assert (vars < v && v <= vars + num_vars);
  return (int)(long)(v - vars);
}

static int map_lit (int lit) {
  Var * v = lit2var (lit);
  int res = v->mapped;
  assert (res);
  if (lit < 0) res = -res;
  return res;
}

static void print_clause (Clause * c, FILE * file) {
  Node * p;
  for (p = c->nodes; p->lit; p++)
    fprintf (file, "%d ", map_lit (p->lit));
  fputs ("0\n", file);
}

static void print_clauses (FILE * file) {
  Clause * p;
  for (p = first_clause; p; p = p->next)
    print_clause (p, file);
}

static void print_scope (Scope * s, FILE * file) {
  Var * p;
  fputc (s->type < 0 ? 'a' : 'e', file);
  for (p = s->first; p; p = p->next) {
    fprintf (file, " %d", map_lit (var2lit (p)));
  }
  fputs (" 0\n", file);
}

static inline int empty_scope (Scope * s) { return s->nr_vars == 0; }


static void print_scopes (FILE * file) {
  Scope * p;
  for (p = outer_most_scope; p; p = p->inner) {
    if (empty_scope (p)) continue;
    print_scope (p, file);
  }
}

static void print (FILE * file) {
  fprintf (file, "p cnf %d %lu\n", mapped, num_clauses);
  print_scopes (file);
  print_clauses (file);
}

static void release_clauses (void) {
  Clause * p, * next;
  size_t bytes;
  for (p = first_clause; p; p = next) {
    next = p->next;
    bytes = bytes_clause (p->orig_size);
    DEC (bytes);
    free (p);
  }
}

static void release_scopes (void) {
  Scope * p, * next;
  for (p = outer_most_scope; p; p = next) {
    next = p->inner;
    DEL (p);
  }
}

static void release (void) {
  release_clauses ();
  release_scopes ();
  DELN (line, szline);
  DELN (lits, size_lits);
  DELN (vars, num_vars + 1);
  assert (getenv ("LEAK") || current_bytes == 0);
}

static void addPrefix (QDPLL *depqbf) {
  Var *v;
  Scope *s;

  for (s = outer_most_scope; s; s = s->inner) {
    qdpll_new_scope (depqbf,
      (s->type < 0 ? QDPLL_QTYPE_FORALL : QDPLL_QTYPE_EXISTS));
    for (v = s->first; v; v = v->next) {
      qdpll_add(depqbf, map_lit(var2lit(v)));
    }
    qdpll_add(depqbf, 0);
  }
}

static void addMatrix(QDPLL *depqbf) {
  Clause *p;
  Node *n;

  for (p = first_clause; p; p = p->next) {
    for (n = p->nodes; n->lit; n++) {
      qdpll_add(depqbf, map_lit(n->lit));
    }
    qdpll_add(depqbf, 0);
  }
}

static QDPLL * init_depqbf () {
  QDPLL *depqbf = qdpll_create();
  qdpll_configure(depqbf, "--dep-man=simple");
  qdpll_configure(depqbf, "--incremental-use");

  addPrefix(depqbf);
  addMatrix(depqbf);

  return depqbf;
}

//---------------------------------------------------------------------------

typedef struct Hashed_clause {
    Clause * c;
    int size;
    UT_hash_handle hh;
} Hashed_clause;

typedef struct Hashed_var {
    Var * v;
    int occs[2];
    int free;
    UT_hash_handle hh;
} Hashed_var;

typedef struct Formula {
    Hashed_clause * clauses;
    Hashed_var * vars;
    Scope * scope;
    Var * first_var;
    int sat_status;
    int vars_left;
} Formula;

typedef struct Mcts_node {
  Formula * state;
  int is_terminal;
  int player;
  int lit;
  struct Mcts_node * parent;
  struct Mcts_node * child0; /* child0 <-> action -1 */
  struct Mcts_node * child1; /* child1 <-> action +1 */
  int num_visits;
  int value;
} Mcts_node;


static Mcts_node * root;
static int MAXITER = 1000;
static double EXPL_CONST = 1.41421356237; // sqrt(2)
static uint64_t bits[16];
static uint64_t MCTS_bits[16];
static int reward, vars_left, rollout_flag, nr_outer_most_vars;
static int * outer_most_vars;
mpz_t paths, one, temp;

#define SETBIT(ByteArr,Bit) ((ByteArr)[Bit/64] |=  (1<<(Bit%64)))
#define CLEARBIT(ByteArr,Bit) ((ByteArr)[Bit/64] &= ~(1<<(Bit%64)))
#define TESTBIT(ByteArr,Bit) ((ByteArr)[Bit/64] & (1<<(Bit%64)))

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL

// Return 64-bit FNV-1a hash for key (NUL-terminated). See description:
// https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
static uint64_t hash_fct(const uint64_t * key) {
    uint64_t hash = FNV_OFFSET;
    for (int i = 0; i < 10; i++) {
        hash ^= (uint64_t) *(key+i);
        hash *= FNV_PRIME;
    }
    return hash;
}

typedef struct Assignment {
    uint64_t key;
    UT_hash_handle hh;
} Assignment;

static Assignment * hashes = NULL;

static inline int rand_assignment (void) {
   return 2*(rand() % 2) - 1;
}

static inline void update_visited_paths (int depth_subtree) {
    mpz_mul_2exp(temp, one, depth_subtree);
    mpz_add(paths,paths,temp);
}

static Formula * init_formula (void){
    Clause * c;
    Formula * f;
    Var * v;
    Scope * s;
    Hashed_clause * hc;
    Hashed_var * hv;
    NEW (f);
    vars_left = 0;
        
    s = outer_most_scope;
    v = s->first;
    while(!v->free){
        if(v->next) v = v->next;
        else {
            s = s->inner;
            v = s->first;
        }
    }
    f->scope = s;
    f->first_var = v;
    f->sat_status = 0;
    
    f->clauses = NULL;
    for (c = first_clause; c; c=c->next){
        NEW (hc);
        hc->c = c;
        hc->size = c->size;
        HASH_ADD_PTR (f->clauses, c, hc);
    }
    
    f->vars = NULL;
    while(s) {
        while(v) {
            if (!v->free) {v = v->next; continue;}
            vars_left++;
            NEW (hv);
            hv->v = v;
            hv->free = 1;
            hv->occs[0] = v->occs->count;
            hv->occs[1] = (v->occs+1)->count;
            HASH_ADD_PTR (f->vars, v, hv);
            v = v->next;
        }
        s = s->inner;
        if (s) v = s->first;
    }
    f->vars_left = vars_left;
    return f;
}

static Hashed_clause * copy_clauses (Hashed_clause * h) {
    Clause * c;
    Hashed_clause * s, * s_new, * h_new;
    h_new = NULL;
    for (s = h; s; s=s->hh.next){
        NEW (s_new);
        c = s->c;
        s_new->c = c;
        s_new->size = s->size;
        HASH_ADD_PTR (h_new, c, s_new);
    }
    return h_new;
}

static Hashed_var * copy_vars (Hashed_var * h) {
    Var * v;
    Hashed_var * s, * s_new, * h_new;
    h_new = NULL;
    for (s = h; s; s=s->hh.next){
        NEW (s_new);
        v = s->v;
        s_new->v = v;
        s_new->occs[0] = s->occs[0];
        s_new->occs[1] = s->occs[1];
        s_new->free = s->free;
        HASH_ADD_PTR (h_new, v, s_new);
    }
    return h_new;
}

static Formula * copy_formula (Formula * f) {
    Formula * f_new;
    NEW (f_new);
    
    f_new->sat_status = f->sat_status;
    f_new->vars_left = f->vars_left;

    if (f_new->sat_status) return f_new;
    
    f_new->clauses = copy_clauses (f->clauses);
    f_new->vars = copy_vars (f->vars);
    f_new->scope = f->scope;
    f_new->first_var = f->first_var;
    
    return f_new;
}

static int get_unit_lit_mcts (Clause * c, Hashed_var * hash) {
    Node * p;
    Var * v;
    Hashed_var * hv;
    int lit;
    for(p = c->nodes; (lit = p->lit); p++) {
        v = lit2var(lit);
        if (!v->free) continue;
        HASH_FIND_PTR (hash, &v, hv);
        if (hv->free) break;
    }
    return lit;
}

static void update_var (Formula * f) {
    Scope * s = f->scope;
    Var * v = f->first_var;
    Hashed_var * f_vars = f->vars;
    Hashed_var * hv;
    
    HASH_FIND_PTR (f_vars, &v, hv);
    assert(!hv->free);
    
    hv = NULL;
    while(!hv || !hv->free) {
        if (v->next) v = v->next;
        else {
            s = s->inner;
            v = s->first;
        }
        HASH_FIND_PTR (f_vars, &v, hv);
    }
    f->scope = s;
    f->first_var = v;
}

static int set_var_mcts (Formula * f, Var * v, int assignment) {
    Var * w;
    Node * n, * p;
    Clause * c;
    Occ * occ;
    Hashed_clause * hc;
    Hashed_clause * f_clauses = f->clauses;
    Hashed_var * hv;
    Hashed_var * f_vars = f->vars;
    int lit;
    
    HASH_FIND_PTR (f_vars, &v, hv);
    if(!hv->free) return 0;
    hv->free = 0;
    if (rollout_flag) vars_left--;
    else f->vars_left--;
                    
    // update the clauses
    // remove all clauses where v becomes True
    occ = v->occs + (assignment < 0);
    
    for (n = occ->first; n; n = n->next){
        c = n->clause;
        HASH_FIND_PTR (f_clauses, &c, hc);
        if(hc) {
            HASH_DEL (f_clauses, hc);
            DEL (hc);
            for(p = c->nodes; (lit = p->lit); p++) {
                w = lit2var(lit);
                if (!w->free) continue;
                HASH_FIND_PTR (f_vars, &w, hv);
                if (!hv->free) continue;
                if (--(hv->occs[lit < 0]) == 0) push_forced (existential(lit) ? -lit : lit);
            }
        }
    }
        
    if (!f_clauses) {
        f->sat_status = SAT;
        f->clauses = f_clauses;
        return 1;
    }
    
    // update all clauses where v becomes False
    occ = v->occs + (assignment > 0);
    for (n = occ->first; n; n = n->next){
        c = n->clause;
        HASH_FIND_PTR (f_clauses, &c, hc);
        if (hc) {
            if((--(hc->size)) == 1) {
                lit = get_unit_lit_mcts (c,f_vars);
                if (existential(lit)) push_forced (lit);
                else{
                    f->sat_status = UNSAT;
                    f->clauses = f_clauses;
                    return 1;
                }
            }
            if (!hc->size) {
                f->sat_status = UNSAT;
                f->clauses = f_clauses;
                return 1;
            }
        }
    }
        
    // update hash
    f->clauses = f_clauses;
    // if we reach this point, then sat of formula is still unknown
    f->sat_status = 0;
    return 1;
}

static void update_formula (Formula * f, int assignment) {
    assert(!f->sat_status);
            
    int lit, success;
    Var * v = f->first_var;
    
    success = set_var_mcts (f,v,assignment);
    if (success && assignment > 0) SETBIT(MCTS_bits,var2lit(v));
    if (f->sat_status) goto DONE;
        
    // set forced literals
    while (forced_lits) {
        lit = pop_forced_lit ();
        v = lit2var(lit);
        success = set_var_mcts (f,v,sign(lit));
        if (success && !rollout_flag) update_visited_paths (num_vars - v->scope_idx);
        if (f->sat_status) goto DONE;
    }
            
    // update first variable and scope if necessary
    update_var (f);
    
    DONE:
        forced_lits = 0;
}


static void release_clauses_mcts (Hashed_clause * hash){
    Hashed_clause * s, * tmp;
    HASH_ITER(hh, hash, s, tmp) {
        HASH_DEL (hash, s);
        DEL (s);
    }
}

static void release_vars_mcts (Hashed_var * hash){
    Hashed_var * s, * tmp;
    HASH_ITER(hh, hash, s, tmp) {
        HASH_DEL (hash, s);
        DEL (s);
    }
}

static void release_formula (Formula * f){
    release_clauses_mcts (f->clauses);
    release_vars_mcts (f->vars);
    DEL (f);
}


static Mcts_node * init_mcts_node (Formula * f) {
    Mcts_node * node;
    NEW (node);
    node->state = f;
    node->is_terminal = f->sat_status;
    node->player = f->scope->type;
    node->lit = var2lit(f->first_var);
    node->num_visits = 0;
    node->value = 0;
    
    return node;
}

static Mcts_node * expand (Mcts_node * parent){
    Mcts_node * child;
    Formula * f;
                        
    if (!parent->child0) {
        f = copy_formula(parent->state);
        update_formula (f,-1);
        child = init_mcts_node (f);
        parent->child0 = child;
    }
    else {
        f = parent->state;
        parent->state = NULL;
        update_formula (f,1);
        child = init_mcts_node (f);
        parent->child1 = child;
    }
    child->parent = parent;
    return child;
}

static Mcts_node * get_best_child (Mcts_node * parent){
        
    float v0, v1;
    int p = parent->player;
    float N = EXPL_CONST * sqrt(log(parent->num_visits));
    Mcts_node * c0 = parent->child0;
    Mcts_node * c1 = parent->child1;
    Mcts_node * best_child;
    int n0 = c0->num_visits;
    int n1 = c1->num_visits;

    v0 = (float) (p * (c0->value)) / n0 + N / sqrt (n0);
    v1 = (float) (p * (c1->value)) / n1 + N / sqrt (n1);
    
    if (v0 == v1)
        best_child = rand() % 2 ? c0 : c1;
    else
        best_child =  v0 > v1 ? c0 : c1;
                
    if (best_child == c1) SETBIT(MCTS_bits,parent->lit);
        
    return best_child;
        
}

static Mcts_node * selection (void) {
    Mcts_node * node = root;
    while (!node->is_terminal) {
        if (node->child0 && node->child1) node = get_best_child (node);
        else return expand (node);
    }
    return node;
}


void print_assignment (uint64_t* arr) {
    for(int j = 0; j < 8; j++) printf("%i ", TESTBIT(arr,j) ? 1 : 0);
    printf("\n");
}


static void clear_assignment (void) {
    for(int i = 0; i < 16; i++)
        MCTS_bits[i] = bits[i];
}

static void save_assignment (void) {
    Assignment * a;
    uint64_t hash = hash_fct(MCTS_bits);
        
    HASH_FIND_INT(hashes, &hash, a);
    if (!a) {
        NEW (a);
        a->key = hash;
        HASH_ADD_INT(hashes,key,a);
        update_visited_paths (vars_left);
    }
}

static int rollout (Formula * f){
    if (f->sat_status) return f->sat_status == SAT ? 1 : -1;
    
    Formula * g = copy_formula (f);
    rollout_flag = 1;
    
    while (!g->sat_status) {
        update_formula(g, rand_assignment());
    }
    release_formula (g);
    rollout_flag = 0;
    
    return g->sat_status == SAT ? 1 : -1;
}

static void backpropagate (int reward, Mcts_node * start) {
    Mcts_node * node = start;
    while (node) {
        node->num_visits++;
        node->value += reward;
        node = node->parent;
    }
}

static void explore (void) {
    Mcts_node * node;
    clear_assignment ();
    node = selection ();
    vars_left = node->state->vars_left;
    reward = rollout (node->state);
    save_assignment ();
    backpropagate (reward,node);
}

static void release_tree (Mcts_node * node) {
    if (!node) return;
    if (node->child0) release_tree (node->child0);
    if (node->child1) release_tree (node->child1);
    if (node->state) release_formula (node->state);
    DEL (node);
}


static int mcts (void){
    assert(MAXITER > 1);
    
    if (!outer_most_scope->first->free) return 1;
    
    int assignment, n0, n1;
    
    for(int i = 0; i < MAXITER; i++)
        explore ();
    
    n0 = root->child0->num_visits;
    n1 = root->child1->num_visits;
    if (n0 == n1) assignment = rand_assignment ();
    else assignment = n0 > n1 ? -1 : 1;
    
    if(!quiet) {
        float r = (float) n0 / (n0 + n1);
        fprintf(ofile, "V %d %.2f 0\n", assignment * root->lit, r);
    }
    
    Mcts_node * old_root = root;
    release_tree (assignment > 0 ? root->child0 : root->child1);
    root = assignment < 0 ? root->child0 : root->child1;
    root->parent = NULL;
    DEL (old_root);
    
    return assignment;
}

//---------------------------------------------------------------------------
static void delete_node (Node * node) {

  int lit = node->lit;
  Occ * occ = lit2occ (lit);
  assert (occ->count > 0);
  if (node->prev) {
    assert (node->prev->next == node);
    node->prev->next = node->next;
  } else {
    assert (occ->first == node);
    occ->first = node->next;
  }
  if (node->next) {
    assert (node->next->prev == node);
    node->next->prev = node->prev;
  } else {
    assert (occ->last == node);
    occ->last = node->prev;
  }
  occ->count--;
  if (!occ->count && lit2var(lit)->free) push_forced(existential(lit) ? -lit : lit);
  
}

static void delete_clause (Clause * clause) {
  size_t bytes;
  int i;
    
  assert (num_clauses > 0);
  bytes = bytes_clause (clause->orig_size);
  if (clause->prev) {
    assert (clause->prev->next == clause);
    clause->prev->next = clause->next;
  } else {
    assert (first_clause == clause);
    first_clause = clause->next;
  }
  if (clause->next) {
    assert (clause->next->prev == clause);
    clause->next->prev = clause->prev;
  } else {
    assert (last_clause == clause);
    last_clause = clause->prev;
  }
  
  for (i = 0; i < clause->orig_size; i++)
    delete_node (clause->nodes + i);

  DEC (bytes);
  free (clause);
  num_clauses--;
}

static int get_unit_lit (Clause * c) {
    Node * p;
    int lit;
    for(p = c->nodes; (lit = p->lit); p++)
        if (lit2var(lit)->free) break;
    return lit;
}

static int set_var (Var * v, int assignment) {
    if (!v->free) return 0;
    v->free = 0;
                
    Node * n, * m;
    Occ * occ;
    int lit = var2lit(v);
    
    // record assignment
    if (assignment > 0) SETBIT(bits,var2lit(v));

         
    // remove all clauses where v becomes True
    occ = v->occs + (assignment < 0);
    for (n = occ->first; n; n = m) {
        m = n->next;
        delete_clause (n->clause);
    }
    if(!num_clauses) {SAT_STATUS = SAT; return 1;}
                
    // update all clauses where v becomes False
    Clause * c;
    occ = v->occs + (assignment > 0);
    for (n = occ->first; n; n = n->next) {
        c = n->clause;
        if (!(--c->size)) {SAT_STATUS = UNSAT; return 1;}
        if (c->size == 1) {
            lit = get_unit_lit (c);
            if (existential(lit)) push_forced (lit);
            else {SAT_STATUS = UNSAT; return 1;}
        }
    }
    
    return 1;
}

static void check_forced_lits (void) {
    int i, lit, success;
    Var * v;
    Occ * occ;
    
    // check for pure lits
    for (i = 1; i <= num_vars; i++) {
        v = vars + i;
        lit = var2lit(v);
        occ = v->occs;
        if (!occ->count && (occ+1)->count) {push_forced(existential(lit) ? -lit : lit); continue;}
        if (occ->count && !(occ+1)->count) push_forced(existential(lit) ? lit : -lit);
    }
    
    // set forced literals
    while (forced_lits) {
        lit = pop_forced_lit ();
        v = lit2var(lit);
        success = set_var (v,sign(lit));
        if (success) update_visited_paths (num_vars - v->scope_idx);
        if (SAT_STATUS) goto DONE;
    }
    
    DONE:
        forced_lits = 0;
}

static void update (int assignment) {
    assert(!SAT_STATUS);
            
    Var * v;
    int lit;
    Scope * s;
        
    // get first free variable
    v = outer_most_scope->first;
    if (v->next)
        outer_most_scope->first = v->next;
    else {
        s = outer_most_scope->inner;
        DEL (outer_most_scope);
        outer_most_scope = s;
    }
    
    if(!v->free) return;
        
    set_var (v, assignment);
    if (SAT_STATUS) goto DONE;
        
    // set forced literals
    while (forced_lits) {
        lit = pop_forced_lit ();
        set_var (lit2var(lit),sign(lit));
        if (SAT_STATUS) goto DONE;
    }
    
    DONE:
        forced_lits = 0;
}

static void play (void){
    check_forced_lits ();
    if (SAT_STATUS) return;
    CORRECT_RESULT = 0;
    Formula * f = init_formula ();
    root = init_mcts_node (f);
    while(!SAT_STATUS) update (mcts ());
}

static void print_coverage (FILE * file) {
    if (quiet) return;
    
    mpf_t cov;
    mpf_init (cov);
    
    if (empty_clause) mpf_set_z (cov, one);
    else {
        mpf_set_z (cov, paths);
        mpf_div_2exp (cov, cov, num_vars);
    }
    gmp_fprintf (file, "Coverage : %.4Ff\n", cov);
}

static void release_mcts (void) {
    Assignment * s, * tmp;
    HASH_ITER(hh, hashes, s, tmp) {
        HASH_DEL (hashes, s);
        DEL (s);
    }
    
    DELN (outer_most_vars,nr_outer_most_vars);
    DELN (forced_stack, size_forced);
    release_tree (root);
}

static void set_assumptions (void) {
    int lit, i;
    for (i = 0; i < nr_outer_most_vars; i++) {
        lit = outer_most_vars[i];
        qdpll_assume(qdpll, TESTBIT(bits,lit) ? lit : -lit);
    }
    return;
}

static void remember_outer_most_vars (void) {
    Scope * s = outer_most_scope;
    Var * v;
    NEWN(outer_most_vars, nr_outer_most_vars);
    int i = 0;
    for (v = s->first; v; v = v->next) outer_most_vars[i++] = v->mapped;
}

//-----------------------------------------------------------------------------

int main (int argc, char ** argv) {
  const char * perr;
  const char *iname, *oname;
  
  ofile = stdout;
  oname = NULL;
  
  int opt;
  quiet = 0;

  while ((opt = getopt(argc, argv, "qo:")) != -1) {
      switch (opt) {
          case 'q':
              quiet = 1;
              break;
          case 'o':
              oname = optarg;
              break;
          default: /* '?' */
              fprintf(stderr, "Usage: %s [-q] [-o path/to/output/file] path/to/input/file\n",argv[0]);
              exit(EXIT_FAILURE);
      }
  }
  if (optind >= argc) {
    fprintf(stderr, "Expected argument after options\n");
    exit(EXIT_FAILURE);
  }
  if (!(2 <= argc && argc <= 5)) die ("invalid number of arguments");
  iname = argv [optind];
  
  ifile = fopen (iname, "r");
  if (!ifile) die ("can not read '%s'", iname);
  if (oname) {
    ofile = fopen(oname, "w");
    if (!ofile) die ("can not open '%s'", oname);
  }
  else ofile = stdout;
  
  NEWN (forced_stack,size_forced);
  memset(bits,0,sizeof(bits));
  memset(MCTS_bits,0,sizeof(MCTS_bits));
  mpz_inits (paths,temp,NULL);
  mpz_init_set_ui (one,(unsigned long)1);
  
  perr = parse ();
  
  int type = outer_most_scope->type;
  nr_outer_most_vars = outer_most_scope->nr_vars;
  remember_outer_most_vars ();
  
  qdpll = init_depqbf ();
  play ();
  print_coverage (ofile);
  
  if (CORRECT_RESULT) goto DONE;
  if ((type > 0 && SAT_STATUS == SAT) || (type < 0 && SAT_STATUS == UNSAT)) {
    set_assumptions ();
    SAT_STATUS = qdpll_sat(qdpll);
    if ((type > 0 && SAT_STATUS == SAT) || (type < 0 && SAT_STATUS == UNSAT))
        goto DONE;
    qdpll_reset (qdpll);
  }
  SAT_STATUS = qdpll_sat(qdpll);
  
  DONE :
  printf ("Result: %s \n", SAT_STATUS == SAT ? "SAT" : "UNSAT");
  
  // clean up
  if (ifclose) {
    fclose (ifile);
    fclose (ofile);
  }
  release_mcts ();
  release ();
  
  return SAT_STATUS;
}

#ifndef NDEBUG
void dump (void) { print (stdout); }
void dump_clause (Clause * c) { print_clause (c, stdout); }
void dump_scope (Scope * s) { print_scope (s, stdout); }
#endif
