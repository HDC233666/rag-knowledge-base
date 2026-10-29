-- MySQL 初始化脚本（docker compose --profile mysql 时自动执行一次）
--
-- 应用启动时也会自己建表（app/db.py 的 DDL），这份脚本的作用是：
--   1) 显式把表结构写出来，方便审阅/评审，不用翻 Python；
--   2) 有 DBA 的团队可以先用它建好库，再让应用连上来。
-- 两份 DDL 必须保持一致；改表结构时两边一起改。

CREATE DATABASE IF NOT EXISTS rag_kb DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE rag_kb;

-- 文档表：一行一个上传的文件；collection 支持多知识库隔离
CREATE TABLE IF NOT EXISTS documents (
    id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name       VARCHAR(255)    NOT NULL,
    collection VARCHAR(64)     NOT NULL DEFAULT 'default',
    chunks     INT             NOT NULL DEFAULT 0,
    chars      INT             NOT NULL DEFAULT 0,
    created_at TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_documents_collection (collection)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 切分块表：外键级联删除，删文档时块一起删
CREATE TABLE IF NOT EXISTS chunks (
    id     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    doc_id BIGINT UNSIGNED NOT NULL,
    idx    INT             NOT NULL,
    text   MEDIUMTEXT      NOT NULL,
    chars  INT             NOT NULL DEFAULT 0,
    PRIMARY KEY (id),
    KEY idx_chunks_doc (doc_id),
    CONSTRAINT fk_chunks_doc FOREIGN KEY (doc_id)
        REFERENCES documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
