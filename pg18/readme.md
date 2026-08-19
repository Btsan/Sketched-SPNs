Cardinality estimates injection patch for Postgres 18.4. Use in place of files with the same names in [the original Postgres 13.1 patch by Han and Wu et al.](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark). Functionality remains the same.

### Docker build example

```bash
bash benchmark_builder.sh
tar cvf postgresql-18.3.tar.gz postgresql-18.3 && mv postgresql-18.3.tar.gz dockerfile/
sudo docker build -t ceb-18 dockerfile/
```